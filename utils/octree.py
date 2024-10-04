import time
import numpy as np
import torch
import torch.nn.functional as F
import torch_scatter

DEBUG_OCTREE = False


def into_box(boxes, x, inv=False):
    """
    [..., 6] X [..., 3] -> [..., 3]
    """
    if inv:
        return x * boxes[..., 3:] + boxes[..., :3]
    return (x - boxes[..., :3]) / boxes[..., 3:]


def inside_box(boxes, x, exactly=True):
    """
    [..., 6] X [..., 3] -> [..., 1]
    """
    if not exactly:
        lt1 = torch.prod((x - boxes[..., :3]) / boxes[..., 3:] <= 1, -1, keepdim=True)
        gt0 = torch.prod((x - boxes[..., :3]) / boxes[..., 3:] >= 0, -1, keepdim=True)
        return lt1 * gt0
    lt1 = torch.prod((x - boxes[..., :3]) / boxes[..., 3:] < 1, -1, keepdim=True)
    gt0 = torch.prod((x - boxes[..., :3]) / boxes[..., 3:] > 0, -1, keepdim=True)
    return lt1 * gt0


def which_oct_cell(boxes, x):
    """
    [..., 6] X [..., 3] -> [..., 1]
    """
    idx = (into_box(boxes, x) * 2).long()
    idx = torch.clip(idx, 0, 1)
    return 4 * idx[..., 0] + 2 * idx[..., 1] + idx[..., 2]


def intersect_box(boxes, rays_o, rays_d, forward_only=True):
    """
    [N?, 3] X [N, 3], [N, 3] -> [N, 1], [N, 1], [N, 1]
    """
    inv_dir = 1.0 / rays_d
    t_min = (boxes[..., :3] - rays_o) * inv_dir
    t_max = (boxes[..., 3:] + boxes[..., :3] - rays_o) * inv_dir
    t1 = torch.minimum(t_min, t_max)
    t2 = torch.maximum(t_min, t_max)

    near = torch.maximum(torch.maximum(t1[..., 0:1], t1[..., 1:2]), t1[..., 2:3])
    far = torch.minimum(torch.minimum(t2[..., 0:1], t2[..., 1:2]), t2[..., 2:3])

    if forward_only:
        return torch.logical_and(near <= far, far >= 0), torch.maximum(near, torch.zeros_like(near)), far

    return near <= far, near, far


def divide(boxes):
    """
    [..., 6] -> [..., 8, 6]
    """
    box_min = boxes[..., None, :3]
    box_size = boxes[..., None, 3:]
    ofs = torch.linspace(0, 7, 8, dtype=torch.long, device=boxes.device).view(-1, 1)
    ofs = torch.cat([torch.div(ofs, 4, rounding_mode='trunc') % 2,
                     torch.div(ofs, 2, rounding_mode='trunc') % 2, ofs % 2], -1)
    new_min = box_min + ofs * box_size / 2
    new_size = (box_size / 2).expand(new_min.shape)

    return torch.cat([new_min, new_size], -1)


class Octree:

    def __init__(self, box_min, box_size, device="cuda"):
        self.boxes = torch.cat([torch.tensor(box_min), torch.tensor(box_size)]).view(1, 6)
        self.non_leaf = torch.zeros(1, 1, dtype=torch.long)
        self.links = torch.zeros(1, 8, dtype=torch.long)
        self.device = device
        if device == "cuda":
            self.cuda()
        self.max_depth = -1
        self.cache_index = None
        self.leaf_size = 1.0
        self.whole_box = self.boxes[0]

    def save(self, file):
        state_dict = {
            "boxes": self.boxes,
            "non_leaf": self.non_leaf,
            "links": self.links,
            "cache_index": self.cache_index,
            "max_depth": self.max_depth,
            "leaf_size": self.leaf_size,
        }
        torch.save(state_dict, file)

    def load(self, file):
        state_dict = torch.load(file)
        self.boxes = state_dict["boxes"]
        self.non_leaf = state_dict["non_leaf"]
        self.links = state_dict["links"]
        self.cache_index = state_dict["cache_index"]
        self.max_depth = state_dict["max_depth"]
        self.leaf_size = state_dict["leaf_size"]

    def cuda(self):
        self.device = "cuda"
        self.boxes = self.boxes.cuda()
        self.non_leaf = self.non_leaf.cuda()
        self.links = self.links.cuda()
        return self

    def add_nodes(self, number):
        new_boxes = torch.zeros(number, 6, device=self.device)
        new_non_leaf = torch.zeros(number, 1, dtype=torch.long, device=self.device)
        new_links = torch.zeros(number, 8, dtype=torch.long, device=self.device)
        self.boxes = torch.cat([self.boxes, new_boxes], 0)
        self.non_leaf = torch.cat([self.non_leaf, new_non_leaf], 0)
        self.links = torch.cat([self.links, new_links], 0)

    def build_base_grid(self, div_fn, max_depth, min_depth, cell_size):
        box = self.whole_box
        cell_num = (box[..., 3:] / cell_size).ceil().long()
        new_size = cell_num * cell_size
        self.whole_box[3:] = new_size

        lsp1 = torch.linspace(0, cell_num[0], cell_num[0] + 1, device=self.boxes.device, dtype=torch.long)[:-1]
        lsp2 = torch.linspace(0, cell_num[1], cell_num[1] + 1, device=self.boxes.device, dtype=torch.long)[:-1]
        lsp3 = torch.linspace(0, cell_num[2], cell_num[2] + 1, device=self.boxes.device, dtype=torch.long)[:-1]
        anchor = torch.stack(torch.meshgrid([lsp1, lsp2, lsp3], indexing="ij"), -1).view(-1, 3)
        index = torch.linspace(0, anchor.shape[0] - 1, anchor.shape[0], device=self.boxes.device, dtype=torch.long)
        init_box_min = anchor / cell_num
        init_box_max = (anchor + 1.0) / cell_num
        init_box_min = into_box(self.whole_box, init_box_min, inv=True)
        init_box_max = into_box(self.whole_box, init_box_max, inv=True)
        init_box = torch.cat([init_box_min, init_box_max - init_box_min], -1)
        self.boxes = init_box
        self.non_leaf = torch.zeros(init_box.shape[0], 1, dtype=torch.long, device=self.boxes.device)
        self.links = -torch.ones(init_box.shape[0], 8, dtype=torch.long, device=self.boxes.device)
        self.cache_index = index.view(*cell_num)
        self.build(div_fn, max_depth, min_depth)

    def build(self, div_fn, max_depth, min_depth):
        starts = [0]
        ends = [self.boxes.shape[0]]

        for i in range(max_depth):
            s = starts[i]
            e = ends[i]
            boxes = self.boxes[s:e]
            if i < min_depth:
                dived = torch.ones_like(boxes[..., 0:1], dtype=torch.long)
            else:
                dived = div_fn(boxes.float())
            if not dived.any():
                break
            k = dived.nonzero()[..., 0]
            n = k.shape[0]
            self.non_leaf[s:e] = dived.view(-1, 1)
            lsp1 = torch.linspace(0, 7, 8, dtype=torch.long, device=self.device)
            lsp2 = torch.linspace(e, e + (n - 1) * 8, n, dtype=torch.long, device=self.device)
            new_links = lsp1[None, :] + lsp2[:, None]
            self.links[s:e][k] = new_links
            self.add_nodes(n * 8)

            self.boxes[e:] = divide(boxes[k]).view(-1, 6)
            starts += [e]
            ends += [e + n * 8]

        if DEBUG_OCTREE:
            assert self.boxes.shape[0] == self.links.shape[0] == self.non_leaf.shape[0]

        self.combine_empty(max_depth)
        # self.cache_index = self.gen_grid_index(max(min_depth, 6))
        self.max_depth = max_depth

        bytes = (self.boxes.numel() + self.boxes.numel() + self.non_leaf.numel() + self.cache_index.numel()) * 4
        print(self.boxes.shape[0], "boxes", bytes // 1024 // 1024, "MB")

    def combine_empty(self, max_depth, not_empty_fn=None):
        leaf_size = self.whole_box[3:] / (2 ** max_depth)

        if not_empty_fn is None:
            def not_empty_fn(bosex):
                return (bosex[..., 3:] < leaf_size + 1e-4).prod(-1).bool()

        has_cell_child = not_empty_fn(self.boxes)
        non_leaf = self.non_leaf[..., 0].bool().clone()

        for i in range(max_depth):
            ptr = self.links[non_leaf]
            not_empty = has_cell_child[ptr].sum(-1).bool()
            has_cell_child[non_leaf] = not_empty

        self.non_leaf[non_leaf] = has_cell_child.long()[..., None][non_leaf]
        self.leaf_size = leaf_size

    def gen_grid_index(self, depth):
        res = 2 ** depth
        lsp = torch.linspace(0, 1.0, res + 1, device=self.boxes.device)[:-1]
        anchor = torch.stack(torch.meshgrid([lsp, lsp, lsp], indexing="ij"), -1).view(-1, 3)
        anchor = anchor + 0.5 / res
        anchor = into_box(self.boxes[0], anchor, inv=True)
        index = self.query(anchor, depth)
        return index.view(res, res, res)

    def get_cache(self, x):
        min_res = self.cache_index.shape
        local_x = into_box(self.whole_box, x)
        idx = torch.split((local_x * torch.tensor(min_res, device=local_x.device)).floor().long(), 1, -1)
        ptr = self.cache_index[idx][..., 0]
        return ptr

    def query(self, x, max_depth=-1, no_cache=False):
        """
        [..., 3] -> [..., 1]
        """
        init_shape = list(x.shape[:-1]) + [-1]
        x = x.view(-1, 3)

        inside_root = inside_box(self.whole_box, x).bool()[..., 0]
        all_ptr = -torch.ones_like(x[..., 0], dtype=torch.long)
        x = x[inside_root]

        if x.numel() == 0:
            return all_ptr

        if not no_cache and self.cache_index is not None:
            min_res = self.cache_index.shape
            local_x = into_box(self.whole_box, x)
            idx = torch.split((local_x * torch.tensor(min_res, device=local_x.device)).floor().long(), 1, -1)
            ptr = self.cache_index[idx][..., 0]

            # assert (self.query(x, 6, True)[..., 0] == ptr).all()

        else:
            ptr = torch.zeros_like(x[..., 0], dtype=torch.long)

        if DEBUG_OCTREE:
            assert inside_box(self.whole_box, x).all()

        k = self.non_leaf[ptr].nonzero()[..., 0]

        while k.numel() > 0:
            if max_depth == 0:
                break
            max_depth -= 1

            # if DEBUG_OCTREE:
            #     assert (k < ptr.shape[0]).all() and (k >= 0).all()
            #     assert (ptr < self.links.shape[0]).all() and (ptr >= 0).all()
            #     assert x.shape[0] == ptr.shape[0]

            boxes = self.boxes[ptr[k]]
            # assert inside_box(boxes, x[k]).all()
            oct_idx = which_oct_cell(boxes, x[k])
            sub_links = torch.gather(self.links[ptr[k]], -1, oct_idx[:, None])[..., 0]
            ptr[k] = sub_links
            k = self.non_leaf[ptr].nonzero()[..., 0]

        all_ptr[inside_root] = ptr
        return all_ptr.view(init_shape)

    def cast(self, rays_o, rays_d, hit_fn, eps=5e-4, return_full=False, fn_use_ptr=False):
        """
            [..., 3], [..., ?, 3] -> first hit box satisfying hit_fn (leaf only)
        """
        init_shape = list(rays_o.shape[:-1]) + [-1]
        if len(rays_d.shape) < len(rays_o.shape):
            rays_d = rays_d[..., None, :]
        rays_o = rays_o.view(-1, 3)
        rays_d = rays_d.expand(init_shape).reshape(-1, 3)

        # assume all intersect
        valid, near, far = intersect_box(self.boxes[0], rays_o, rays_d)

        k = valid[..., 0]
        t = near + eps
        t[~k] = -1

        ptr = -torch.ones_like(k, dtype=torch.long)
        pos = torch.zeros_like(rays_o)
        pos[k] = rays_o[k] + t[k] * rays_d[k]
        if k.any():
            ptr[k] = self.query(pos[k])[..., 0]
        k = ptr >= 0

        min_res = torch.tensor(self.cache_index.shape, device=k.device)

        # ii = 0
        # import time
        # st = [time.time()]
        #
        # def tick(msg):
        #     torch.cuda.synchronize()
        #     print(msg, time.time() - st[0])
        #     st[0] = time.time()

        while k.any():
            pos_k = pos[k]
            rays_d_k = rays_d[k]
            boxes_k = self.boxes[ptr[k]]
            # valid, near, far = intersect_box(self.boxes[ptr[k]], pos_k, rays_d_k)

            # extend intersect box
            t_min = (boxes_k[..., :3] - pos_k) / rays_d_k
            t_max = (boxes_k[..., 3:] + boxes_k[..., :3] - pos_k) / rays_d_k
            far = torch.maximum(t_min, t_max).min(dim=-1, keepdim=True)[0]

            # if DEBUG_OCTREE:
            #     if (near != 0).any():
            #         tmp = near != 0
            #         assert (near[tmp].abs() < 1e-4).all()
            #         near[tmp] = 0
            #     if (~valid).any():
            #         assert (far[~valid[..., 0]].abs() < 1e-4).all()
            #         far[~valid[..., 0]] = near[~valid[..., 0]]

            # tick("start")
            t[k] += far + eps
            pos[k] = rays_o[k] + t[k] * rays_d_k

            isd = torch.ones_like(k)
            isd[k] = inside_box(self.boxes[0], pos[k]).bool()[..., 0]
            k = torch.logical_and(k, isd)
            ptr[~isd] = -1

            if k.any():
                # ptr[k] = self.query(pos[k])[..., 0]

                # tick("pre")

                # extend query
                q_x = pos[k]
                q_idx = torch.split((into_box(self.boxes[0], q_x) * min_res).floor().long(), 1, -1)
                q_ptr = self.cache_index[q_idx][..., 0]
                q_k = self.non_leaf[q_ptr].bool()[..., 0]

                while q_k.any():
                    q_ptr_k = q_ptr[q_k]
                    q_ptr[q_k] = \
                        torch.gather(self.links[q_ptr_k], -1, which_oct_cell(self.boxes[q_ptr_k], q_x[q_k])[:, None])[
                            ..., 0]
                    q_k = self.non_leaf[q_ptr].bool()[..., 0]

                ptr[k] = q_ptr
                # tick("query")

                k = ptr > 0
            else:
                break

            # ii += 1
            # if ii >= 1024:
            #     break

            if k.any():
                hit = hit_fn(ptr[k] if fn_use_ptr else self.boxes[ptr[k]].float()).bool()
                # tick("hit")
                k[k.clone()] = ~hit
                # tick("end")
            else:
                break

        # print(ii)
        if return_full:
            return t.reshape(init_shape), ptr.reshape(init_shape), pos.reshape(init_shape)

        return t.reshape(init_shape)


class OctreeSDF:

    def __init__(self, sdf_fn, bounds, thr=0.5, max_iter=-1):
        box = bounds[0], [bounds[1][i] - bounds[0][i] for i in range(3)]
        self.octree = Octree(*box)

        def div_fn(boxes):
            with torch.no_grad():
                size = boxes[..., 3:]
                center = boxes[..., :3] + size * 0.5
                return sdf_fn(center).abs() < size.norm(dim=-1) * thr

        cell_size = 0.05
        self.octree.build_base_grid(div_fn, 4, 0, cell_size)
        leaf_size = torch.ones_like(self.octree.whole_box[3:]) * cell_size / 2 ** 4
        centers = self.octree.boxes[..., :3] + self.octree.boxes[..., 3:] * 0.5

        chunk = 8192
        res = []
        for j in range(0, centers.shape[0], chunk):
            res.append(prox_gradients(sdf_fn, centers[j:j + chunk].float(), leaf_size.min().item() * 2).detach())
        self.sdf_grad = torch.cat(res, 0)
        self.sdf_grad = self.sdf_grad / torch.clamp(torch.norm(self.sdf_grad, dim=-1, keepdim=True), min=1e-4)
        res = []
        for j in range(0, centers.shape[0], chunk):
            res.append(sdf_fn(centers[j:j + chunk].float()).detach())
        self.sdf_val = torch.cat(res, 0)
        self.centers = centers

        self.min_step = leaf_size.min().item() + 1e-4

        is_leaf = (self.octree.boxes[..., 3:] < self.min_step).prod(-1).bool()
        is_surface = torch.relu(self.sdf_val) <= 1e-4
        self.hit_ptr = is_surface
        self.max_iter = max_iter

    def hit(self, ptr):
        return self.hit_ptr[ptr]

    def normal(self, point):
        ptr = self.octree.query(point)[..., 0]
        valid = ptr >= 0
        normal = torch.ones_like(point)
        normal[valid] = self.sdf_grad[ptr[valid]]
        return normal

    def cast(self, rays_o, rays_d, return_is_hit=False):
        t, ptr, x = self.multi_step_cast(rays_o, rays_d, return_full=True)
        ptr = ptr[..., 0]
        t = t[..., 0]
        valid = ptr >= 0

        if valid.any():
            normal = self.sdf_grad[ptr[valid]]
            point = self.centers[ptr[valid]] - normal * self.sdf_val[ptr[valid]].view(-1, 1)
            dist = ((point - x[valid]) * normal).sum(-1)
            speed = (rays_d[valid] * normal).sum(-1)
            speed[speed == 0] = 1e-4
            dt = torch.clamp(dist / speed, -self.min_step * 10, self.min_step * 10)
            t[valid] += dt

        if return_is_hit:
            return t[..., None], valid
        return t[..., None]

    def volume_render(self, rays_o, rays_d):
        step_size = 1e-3
        n_samp = 20
        t = torch.linspace(0, 1, n_samp + 1, device=rays_o.device) * n_samp * step_size
        t_mid = (t[1:] + t[:-1]) / 2

        sample_pts = rays_o[:, None, :] + rays_d[:, None, :] * t[:, None]
        ptr = self.octree.query(sample_pts.reshape(-1, 3))
        sdf = self.sdf_val[ptr].view(-1, n_samp + 1)

        sigma = F.softplus(sdf * np.exp(3.0))
        sigma = F.relu(sigma[:, 1:] - sigma[:, :-1])
        alpha = 1.0 - torch.exp(-sigma)
        weights = alpha * torch.cumprod(torch.cat([torch.ones_like(alpha[:, :1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]
        weights_sum = weights.sum(dim=-1)
        depth = (t_mid * weights).sum(dim=-1) + (1 - weights_sum) * n_samp * step_size

        return depth

    def fast_volume_render(self, rays_o, rays_d, n_samp=100, step_size=1e-3):
        t = torch.linspace(0, 1, n_samp + 1, device=rays_o.device) * n_samp * step_size
        t = t + step_size
        t_mid = t[1:]

        sample_pts = rays_o[:, None, :] + rays_d[:, None, :] * t_mid[:, None]
        ptr = self.octree.query(sample_pts.reshape(-1, 3))
        sdf = self.sdf_val[ptr].view(-1, n_samp)

        hit = sdf <= step_size
        idx = first_nonzero(hit)

        return t[idx]

    def query(self, q_x):
        min_res = torch.tensor(self.octree.cache_index.shape, device=q_x.device)
        # extend query
        q_idx = torch.split(
            ((q_x - self.octree.whole_box[..., :3]) / self.octree.whole_box[..., 3:] * min_res).floor().long(), 1, -1)
        q_ptr = self.octree.cache_index[q_idx][..., 0]
        q_k = self.octree.non_leaf[q_ptr].bool()[..., 0]

        while q_k.any():
            q_ptr_k = q_ptr[q_k]
            tmp_boxes = self.octree.boxes[q_ptr_k]

            idx = torch.clip(((q_x[q_k] - tmp_boxes[..., :3]) / tmp_boxes[..., 3:] * 2).long(), 0, 1)
            oct_cell = 4 * idx[..., 0] + 2 * idx[..., 1] + idx[..., 2]

            q_ptr[q_k] = torch.gather(self.octree.links[q_ptr_k], -1, oct_cell[:, None])[..., 0]
            q_k = self.octree.non_leaf[q_ptr].bool()[..., 0]

        return q_ptr

    def multi_step_cast(self, rays_o, rays_d, eps=1e-3, return_full=False):
        """
            [..., 3], [..., ?, 3] -> first hit box satisfying hit_fn (leaf only)
        """
        init_shape = list(rays_o.shape[:-1]) + [-1]
        if len(rays_d.shape) < len(rays_o.shape):
            rays_d = rays_d[..., None, :]
        rays_o = rays_o.view(-1, 3)
        rays_d = rays_d.expand(init_shape).reshape(-1, 3)

        # if use early stopping, bias the start of the rays
        if self.max_iter > 0:
            rays_o = rays_o + rays_d * 0.005

        # assume all intersect
        valid, near, final_far = intersect_box(self.octree.whole_box, rays_o, rays_d)

        k = valid[..., 0]
        t = near + eps
        t[~k] = -1

        ptr = -torch.ones_like(k, dtype=torch.long)
        pos = torch.zeros_like(rays_o)
        pos[k] = rays_o[k] + t[k] * rays_d[k]
        if k.any():
            ptr[k] = self.octree.query(pos[k])[..., 0]
        k = ptr >= 0

        ii = 0
        jj = 0
        final_k = None

        while k.any():
            if self.max_iter > 0 and ii > self.max_iter:
                break
            pos_k = pos[k]
            rays_d_k = rays_d[k]
            boxes_k = self.octree.boxes[ptr[k]]

            # if not inside_box(boxes_k, pos_k, False).all():
            #     _, _, far_ = intersect_box(self.octree.boxes[ptr[k]], pos[k], rays_d[k])
            #     assert far_.min() > -0.005

            # extend intersect box
            valid, near, far = intersect_box(boxes_k, pos_k, rays_d_k)

            # if (far < 0).any():
            #     assert far.min() > -0.005

            step_size = 0.001
            if self.max_iter > 0:
                step_size = 0.005
                if k.numel() > 100000:
                    step_size = 0.01
            multi_samp = int(np.clip(k.numel() * 10, 1, 2000000) // k.sum())
            multi_samp = np.clip(multi_samp, 1, 100)
            small_step = (far < multi_samp * step_size)[..., 0]

            new_far = self.fast_volume_render(pos_k[small_step], rays_d_k[small_step], multi_samp, step_size)
            far[small_step] = new_far[..., None]

            t[k] += far + eps
            pos[k] = rays_o[k] + t[k] * rays_d_k

            isd = torch.ones_like(k)
            isd[k] = inside_box(self.octree.whole_box, pos[k]).bool()[..., 0]
            k = torch.logical_and(k, isd)
            ptr[~isd] = -1

            if k.any():
                ptr[k] = self.octree.query(pos[k])[..., 0]
                k = ptr >= 0  # this bug cost me 2 hours to fix !!!
                # if not inside_box(self.octree.boxes[ptr[k]], pos[k], False).all():
                #     _, _, far_ = intersect_box(self.octree.boxes[ptr[k]], pos[k], rays_d[k])
                #     if far_.min() < -0.005:
                #         print("[Neg Far]", far_.min())
                #     assert far_.min() > -0.005

            if k.any():
                hit = self.hit_ptr[ptr[k]]
                k[k.clone()] = ~hit

            ii += 1
            if k.sum() / k.numel() <= 0.01 and jj == 0:
                jj = ii

        if jj == 0:
            print(ii, "and", jj, "with", k.sum() / (k.numel() + 1e-8), "in", k.numel())

        if return_full:
            return t.reshape(init_shape), ptr.reshape(init_shape), pos.reshape(init_shape)

        return t.reshape(init_shape)


def first_nonzero(values):
    values = torch.cat([values, torch.ones_like(values[:, -1:])], -1)
    seg, idx = values.nonzero().permute(1, 0)
    res = torch_scatter.scatter_min(idx, seg)
    return res[0]


def prox_gradients(func, x, dx, diff=False):
    if diff:
        y0 = func(x)[..., None]
        grads = []
        for i in range(x.shape[-1]):
            ofs = torch.zeros_like(x)
            ofs[..., i] = dx
            y1 = func(x + ofs)[..., None]
            grads.append((y1 - y0) / dx)
        return torch.cat(grads, -1)
    not_eval = torch.is_grad_enabled()
    with torch.enable_grad():
        x.requires_grad_(True)
        y = func(x)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=not_eval,
            retain_graph=not_eval,
            only_inputs=True)[0]
    return gradients


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    torch.random.manual_seed(0)
    np.random.seed(0)

    boxes = [[0, 0, 0, 2, 2, 2], [1, 1, 1, 2, 2, 2.]]
    rays_o = [[4, 4, 4], [3, 3, 4.]]
    rays_d = [[-1, -1, -1], [-1, -1, -1.]]
    v, n, f = intersect_box(torch.tensor(boxes), torch.tensor(rays_o), torch.tensor(rays_d))
    print(v, n, f)


    def draw_boxes(boxes):
        for b in boxes:
            b = b.cpu()
            mx, my, sx, sy = b[..., 0], b[..., 1], b[..., 3], b[..., 4]
            # xs = torch.cat([mx, mx + sx, mx + sx, mx, mx], dim=-1)
            # ys = torch.cat([my, my, my + sy, my + sy, my], dim=-1)
            xs = np.array([mx, mx + sx, mx + sx, mx, mx])
            ys = np.array([my, my, my + sy, my + sy, my])
            plt.plot(xs, ys)


    # torch.manual_seed(1)
    #
    # # octree = Octree([0, 0, 0], [1, 1, 1])
    # # octree.build(tst_fn, 5, 3)
    # # x = torch.rand(8024, 3)
    # # # x = torch.tensor([[0.3, 0.7, 0.3]])
    # # ptr = octree.query(x)
    # # # draw_boxes(octree.boxes[ptr.flatten()])
    # # # plt.show()
    # #
    #
    octree = Octree([0, 0, 0], [1, 1, 1])

    out = octree.query(torch.tensor([10, 10, 10], device="cuda"))
    print(out)
    #
    # st = time.time()
    # for i in range(1000):
    #     x = torch.rand(1024 * 256, 3).cuda()
    #     ptr = octree.query(x)
    # print(time.time() - st)
    #
    # res = tst_fn(octree.boxes[ptr]) * 100 + 1
    # stride = 1
    # plt.scatter(x[::stride, 0].cpu(), x[::stride, 1].cpu(), s=0.01, c=res[::stride].cpu())
    # plt.show()
