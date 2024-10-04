import gc


def summary():
    # Log training summaries. This is put behind a host_id check because in
    # multi-host evaluation, all hosts need to run inference even though we
    # only use host 0 to record results.
    if jax.host_id() == 0:
        if step % config.print_every == 0:
            summary_writer.scalar('num_params', num_params, step)
            summary_writer.scalar('train_loss', stats.loss[0], step)
            summary_writer.scalar('train_psnr', stats.psnr[0], step)
            for i, l in enumerate(stats.losses[0]):
                summary_writer.scalar(f'train_losses_{i}', l, step)
            for i, p in enumerate(stats.psnrs[0]):
                summary_writer.scalar(f'train_psnrs_{i}', p, step)
            summary_writer.scalar('weight_l2', stats.weight_l2[0], step)
            avg_loss = np.mean(np.concatenate([s.loss for s in stats_trace]))
            avg_psnr = np.mean(np.concatenate([s.psnr for s in stats_trace]))
            max_grad_norm = np.max(
                np.concatenate([s.grad_norm for s in stats_trace]))
            avg_grad_norm = np.mean(
                np.concatenate([s.grad_norm for s in stats_trace]))
            max_clipped_grad_norm = np.max(
                np.concatenate([s.grad_norm_clipped for s in stats_trace]))
            max_grad_max = np.max(
                np.concatenate([s.grad_abs_max for s in stats_trace]))
            stats_trace = []
            summary_writer.scalar('train_avg_loss', avg_loss, step)
            summary_writer.scalar('train_avg_psnr', avg_psnr, step)
            summary_writer.scalar('train_max_grad_norm', max_grad_norm, step)
            summary_writer.scalar('train_avg_grad_norm', avg_grad_norm, step)
            summary_writer.scalar('train_max_clipped_grad_norm',
                                  max_clipped_grad_norm, step)
            summary_writer.scalar('train_max_grad_max', max_grad_max, step)
            summary_writer.scalar('learning_rate', lr, step)
            steps_per_sec = config.print_every / (time.time() - t_loop_start)
            reset_timer = True
            rays_per_sec = config.batch_size * steps_per_sec
            summary_writer.scalar('train_steps_per_sec', steps_per_sec, step)
            summary_writer.scalar('train_rays_per_sec', rays_per_sec, step)
            precision = int(np.ceil(np.log10(config.max_steps))) + 1
            print(('{:' + '{:d}'.format(precision) + 'd}').format(step) +
                  f'/{config.max_steps:d}: ' + f'i_loss={stats.loss[0]:0.4f}, ' +
                  f'avg_loss={avg_loss:0.4f}, ' +
                  f'weight_l2={stats.weight_l2[0]:0.2e}, ' + f'lr={lr:0.2e}, ' +
                  f'{rays_per_sec:0.0f} rays/sec')
        if step % config.save_every == 0:
            state_to_save = jax.device_get(jax.tree_map(lambda x: x[0], state))
            checkpoints.save_checkpoint(
                FLAGS.train_dir, state_to_save, int(step), keep=100)


def train():
    # Prepare raybatch tensor if batching random rays
    N_rand = args.N_rand
    use_batching = not args.no_batching
    if use_batching:
        # For random ray batching
        print('get rays')
        rays = np.stack([get_rays_np(H, W, K, p) for p in poses[:, :3, :4]], 0)  # [N, ro+rd, H, W, 3]
        print('done, concats')
        rays_rgb = np.concatenate([rays, images[:, None]], 1)  # [N, ro+rd+rgb, H, W, 3]
        rays_rgb = np.transpose(rays_rgb, [0, 2, 3, 1, 4])  # [N, H, W, ro+rd+rgb, 3]
        rays_rgb = np.stack([rays_rgb[i] for i in i_train], 0)  # train images only
        rays_rgb = np.reshape(rays_rgb, [-1, 3, 3])  # [(N-1)*H*W, ro+rd+rgb, 3]
        rays_rgb = rays_rgb.astype(np.float32)
        print('shuffle rays')
        np.random.shuffle(rays_rgb)

        print('done')
        i_batch = 0

    # Move training data to GPU
    if use_batching:
        images = torch.Tensor(images).to(device)
    poses = torch.Tensor(poses).to(device)
    if use_batching:
        rays_rgb = torch.Tensor(rays_rgb).to(device)

    N_iters = 200000 + 1
    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)

    # Summary writers
    # writer = SummaryWriter(os.path.join(basedir, 'summaries', expname))

    start = start + 1
    for i in trange(start, N_iters):
        time0 = time.time()

        # Sample random ray batch
        if use_batching:
            # Random over all images
            batch = rays_rgb[i_batch:i_batch + N_rand]  # [B, 2+1, 3*?]
            batch = torch.transpose(batch, 0, 1)
            batch_rays, target_s = batch[:2], batch[2]

            i_batch += N_rand
            if i_batch >= rays_rgb.shape[0]:
                print("Shuffle data after an epoch!")
                rand_idx = torch.randperm(rays_rgb.shape[0])
                rays_rgb = rays_rgb[rand_idx]
                i_batch = 0

        else:
            # Random from one image
            img_i = np.random.choice(i_train)
            target = images[img_i]
            target = torch.Tensor(target).to(device)
            pose = poses[img_i, :3, :4]

            if N_rand is not None:
                rays_o, rays_d = get_rays(H, W, K, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)

                if i < args.precrop_iters:
                    dH = int(H // 2 * args.precrop_frac)
                    dW = int(W // 2 * args.precrop_frac)
                    coords = torch.stack(
                        torch.meshgrid(
                            torch.linspace(H // 2 - dH, H // 2 + dH - 1, 2 * dH),
                            torch.linspace(W // 2 - dW, W // 2 + dW - 1, 2 * dW)
                        ), -1)
                    if i == start:
                        print(
                            f"[Config] Center cropping of size {2 * dH} x {2 * dW} is enabled until iter {args.precrop_iters}")
                else:
                    coords = torch.stack(torch.meshgrid(torch.linspace(0, H - 1, H), torch.linspace(0, W - 1, W)),
                                         -1)  # (H, W, 2)

                coords = torch.reshape(coords, [-1, 2])  # (H * W, 2)
                select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
                select_coords = coords[select_inds].long()  # (N_rand, 2)
                rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                batch_rays = torch.stack([rays_o, rays_d], 0)
                target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)

        cam_up, cam_look = get_axis(torch.Tensor(pose), batch_rays.device)

        """ Annealing """
        middle = (far + near) / 2
        render_kwargs_train['near'] = middle + (near - middle) * min(max(global_step / 5000, 0.5), 1)
        render_kwargs_train['far'] = middle + (far - middle) * min(max(global_step / 5000, 0.5), 1)

        """ Closure Losses """
        closure_losses = {
            "smooth": 0,
        }

        render_kwargs_train['closure'] = closure_losses
        render_kwargs_train['global_step'] = global_step

        #####  Core optimization loop  #####
        rgb, disp, acc, extras = render(H, W, K, chunk=args.chunk, rays=batch_rays,
                                        verbose=i < 10, retraw=True,
                                        cam_up=cam_up, cam_look=cam_look,
                                        **render_kwargs_train)

        optimizer.zero_grad()
        img_loss = img2mse(rgb, target_s)
        trans = extras['raw'][..., -1]
        loss = img_loss
        psnr = mse2psnr(img_loss)
        if 'rgb0' in extras:
            img_loss0 = img2mse(extras['rgb0'], target_s)
            loss = loss + img_loss0
            psnr0 = mse2psnr(img_loss0)

        if args.tv_loss is not None:
            loss = loss + args.tv_loss * extras['tv_loss']

        for k in closure_losses:
            loss = loss + closure_losses[k]

        loss.backward()

        # nn.utils.clip_grad_norm_(render_kwargs_train['network_fine'].parameters(), 0.1)

        optimizer.step()

        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        ################################

        dt = time.time() - time0
        # print(f"Step: {global_step}, Loss: {loss}, Time: {dt}")
        #####           end            #####

        # Rest is logging
        if i % args.i_weights == 0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            torch.save({
                'global_step': global_step,
                'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict() if render_kwargs_train[
                                                                                                   'network_fine'] is not None else None,
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            print('Saved checkpoints at', path)

        if i % args.i_video == 0 and i > 0:
            # Turn on testing mode
            with torch.no_grad():
                rgbs, disps = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test)
            print('Done, saving', rgbs.shape, disps.shape)
            moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))
            imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / np.max(disps)), fps=30, quality=8)

            # if args.use_viewdirs:
            #     render_kwargs_test['c2w_staticcam'] = render_poses[0][:3,:4]
            #     with torch.no_grad():
            #         rgbs_still, _ = render_path(render_poses, hwf, args.chunk, render_kwargs_test)
            #     render_kwargs_test['c2w_staticcam'] = None
            #     imageio.mimwrite(moviebase + 'rgb_still.mp4', to8b(rgbs_still), fps=30, quality=8)

        if i % args.i_testset == 0 and i > 0:
            testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', poses[i_test].shape)
            with torch.no_grad():
                render_path(torch.Tensor(poses[i_test]).to(device), hwf, K, args.chunk, render_kwargs_test,
                            gt_imgs=images[i_test], savedir=testsavedir)
            print('Saved test set')

        if i % args.i_print == 0:
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}")

        global_step += 1


def test():
    # Test-set evaluation.
    if FLAGS.render_every > 0 and step % FLAGS.render_every == 0:
        # We reuse the same random number generator from the optimization step
        # here on purpose so that the visualization matches what happened in
        # training.
        t_eval_start = time.time()
        eval_variables = jax.device_get(jax.tree_map(lambda x: x[0],
                                                     state)).optimizer.target
        test_case = next(test_dataset)
        pred_color, pred_distance, pred_acc = models.render_image(
            functools.partial(render_eval_pfn, eval_variables),
            test_case['rays'],
            keys[0],
            chunk=FLAGS.chunk)

        vis_suite = vis.visualize_suite(pred_distance, pred_acc)

    # Log eval summaries on host 0.
    if jax.host_id() == 0:
        psnr = math.mse_to_psnr(((pred_color - test_case['pixels']) ** 2).mean())
        ssim = ssim_fn(pred_color, test_case['pixels'])
        eval_time = time.time() - t_eval_start
        num_rays = jnp.prod(jnp.array(test_case['rays'].directions.shape[:-1]))
        rays_per_sec = num_rays / eval_time
        summary_writer.scalar('test_rays_per_sec', rays_per_sec, step)
        print(f'Eval {step}: {eval_time:0.3f}s., {rays_per_sec:0.0f} rays/sec')
        summary_writer.scalar('test_psnr', psnr, step)
        summary_writer.scalar('test_ssim', ssim, step)
        summary_writer.image('test_pred_color', pred_color, step)
        for k, v in vis_suite.items():
            summary_writer.image('test_pred_' + k, v, step)
        summary_writer.image('test_pred_acc', pred_acc, step)
        summary_writer.image('test_target', test_case['pixels'], step)


def ckpt():
    if config.max_steps % config.save_every != 0:
        state = jax.device_get(jax.tree_map(lambda x: x[0], state))
        checkpoints.save_checkpoint(
            FLAGS.train_dir, state, int(config.max_steps), keep=100)


def train_step(model, config, rng, state, batch, lr):
    def loss_fn(variables):

        def tree_sum_fn(fn):
            return jax.tree_util.tree_reduce(
                lambda x, y: x + fn(y), variables, initializer=0)

        weight_l2 = config.weight_decay_mult * (
                tree_sum_fn(lambda z: jnp.sum(z ** 2)) /
                tree_sum_fn(lambda z: jnp.prod(jnp.array(z.shape))))

        ret = model.apply(
            variables,
            key,
            batch['rays'],
            randomized=config.randomized,
            white_bkgd=config.white_bkgd)

        mask = batch['rays'].lossmult
        if config.disable_multiscale_loss:
            mask = jnp.ones_like(mask)

        losses = []
        for (rgb, _, _) in ret:
            losses.append(
                (mask * (rgb - batch['pixels'][..., :3]) ** 2).sum() / mask.sum())
        losses = jnp.array(losses)

        loss = (
                config.coarse_loss_mult * jnp.sum(losses[:-1]) + losses[-1] + weight_l2)

        stats = utils.Stats(
            loss=loss,
            losses=losses,
            weight_l2=weight_l2,
            psnr=0.0,
            psnrs=0.0,
            grad_norm=0.0,
            grad_abs_max=0.0,
            grad_norm_clipped=0.0,
        )
        return loss, stats

    (_, stats), grad = (
        jax.value_and_grad(loss_fn, has_aux=True)(state.optimizer.target))
    grad = jax.lax.pmean(grad, axis_name='batch')
    stats = jax.lax.pmean(stats, axis_name='batch')

    def tree_norm(tree):
        return jnp.sqrt(
            jax.tree_util.tree_reduce(
                lambda x, y: x + jnp.sum(y ** 2), tree, initializer=0))

    if config.grad_max_val > 0:
        clip_fn = lambda z: jnp.clip(z, -config.grad_max_val, config.grad_max_val)
        grad = jax.tree_util.tree_map(clip_fn, grad)

    grad_abs_max = jax.tree_util.tree_reduce(
        lambda x, y: jnp.maximum(x, jnp.max(jnp.abs(y))), grad, initializer=0)

    grad_norm = tree_norm(grad)
    if config.grad_max_norm > 0:
        mult = jnp.minimum(1, config.grad_max_norm / (1e-7 + grad_norm))
        grad = jax.tree_util.tree_map(lambda z: mult * z, grad)
    grad_norm_clipped = tree_norm(grad)

    new_optimizer = state.optimizer.apply_gradient(grad, learning_rate=lr)
    new_state = state.replace(optimizer=new_optimizer)

    psnrs = math.mse_to_psnr(stats.losses)
    stats = utils.Stats(
        loss=stats.loss,
        losses=stats.losses,
        weight_l2=stats.weight_l2,
        psnr=psnrs[-1],
        psnrs=psnrs,
        grad_norm=grad_norm,
        grad_abs_max=grad_abs_max,
        grad_norm_clipped=grad_norm_clipped,
    )

    return new_state, stats, rng


from dataclasses import dataclass, fields as datafields


def dataclass_from_dict(klass, dikt):
    try:
        fieldtypes = {f.name: f.type for f in datafields(klass)}
        return klass(**{f: dataclass_from_dict(fieldtypes[f], dikt[f]) for f in dikt})
    except:
        return dikt


def mip_train():

    dataset = datasets.get_dataset('train', FLAGS.data_dir, config)
    test_dataset = datasets.get_dataset('test', FLAGS.data_dir, config)

    rng, key = random.split(rng)
    model, variables = models.construct_mipnerf(key, dataset.peek())
    num_params = jax.tree_util.tree_reduce(
        lambda x, y: x + jnp.prod(jnp.array(y.shape)), variables, initializer=0)
    print(f'Number of parameters being optimized: {num_params}')
    optimizer = flax.optim.Adam(config.lr_init).create(variables)
    state = utils.TrainState(optimizer=optimizer)
    del optimizer, variables

    learning_rate_fn = functools.partial(
        math.learning_rate_decay,
        lr_init=config.lr_init,
        lr_final=config.lr_final,
        max_steps=config.max_steps,
        lr_delay_steps=config.lr_delay_steps,
        lr_delay_mult=config.lr_delay_mult)

    train_pstep = jax.pmap(
        functools.partial(train_step, model, config),
        axis_name='batch',
        in_axes=(0, 0, 0, None),
        donate_argnums=(2,))

    # Because this is only used for test set rendering, we disable randomization.
    def render_eval_fn(variables, _, rays):
        return jax.lax.all_gather(
            model.apply(
                variables,
                random.PRNGKey(0),  # Unused.
                rays,
                randomized=False,
                white_bkgd=config.white_bkgd),
            axis_name='batch')

    render_eval_pfn = jax.pmap(
        render_eval_fn,
        in_axes=(None, None, 0),  # Only distribute the data input.
        donate_argnums=(2,),
        axis_name='batch',
    )

    ssim_fn = jax.jit(functools.partial(math.compute_ssim, max_val=1.))

    if not utils.isdir(FLAGS.train_dir):
        utils.makedirs(FLAGS.train_dir)
    state = checkpoints.restore_checkpoint(FLAGS.train_dir, state)
    # Resume training a the step of the last checkpoint.
    init_step = state.optimizer.state.step + 1
    state = flax.jax_utils.replicate(state)

    summary_writer = tensorboard.SummaryWriter(FLAGS.train_dir)

    # Prefetch_buffer_size = 3 x batch_size
    gc.disable()  # Disable automatic garbage collection for efficiency.
    stats_trace = []
    reset_timer = True
    for step, batch in zip(range(init_step, config.max_steps + 1), pdataset):
        if reset_timer:
            t_loop_start = time.time()
            reset_timer = False
        lr = learning_rate_fn(step)
        state, stats, keys = train_pstep(keys, state, batch, lr)
        if jax.host_id() == 0:
            stats_trace.append(stats)
        if step % config.gc_every == 0:
            gc.collect()


def eval():
    # Short circuit if only rendering out from trained model
    if args.render_only:
        print('RENDER ONLY')
        with torch.no_grad():
            if args.render_test:
                # render_test switches to test poses
                images = images[i_test]
            else:
                # Default is smoother render_poses path
                images = None

            testsavedir = os.path.join(basedir, expname,
                                       'renderonly_{}_{:06d}'.format('test' if args.render_test else 'path', start))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', render_poses.shape)

            rgbs, _ = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test, gt_imgs=images,
                                  savedir=testsavedir, render_factor=args.render_factor)
            print('Done rendering', testsavedir)
            imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), to8b(rgbs), fps=30, quality=8)

            return
