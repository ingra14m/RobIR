import os
import numpy as np
import trimesh
import imageio
import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileShader, compileProgram
from contextlib import contextmanager


class Shader:

    def __init__(self, vs_or_path, fs_or_path):
        def load_shader(shader_file):
            with open(shader_file) as f:
                shader_source = f.read()
            f.close()
            return str.encode(shader_source)

        vert_shader = load_shader(vs_or_path) if os.path.exists(vs_or_path) else vs_or_path
        frag_shader = load_shader(fs_or_path) if os.path.exists(fs_or_path) else fs_or_path

        self.vs = compileShader(vert_shader, GL_VERTEX_SHADER)
        self.fs = compileShader(frag_shader, GL_FRAGMENT_SHADER)
        self.program = compileProgram(self.vs, self.fs)
        self.use()

    def use(self):
        glUseProgram(self.program)

    def release(self):
        glDeleteShader(self.vs)
        glDeleteShader(self.fs)


class Buffer:

    def __init__(self):
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)

        self.ebo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)

    def bind(self):
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)

    def draw(self, va=None, ea=None, ele=GL_TRIANGLES):
        self.bind()
        if va is not None:
            va = va.astype(np.float32)
            glBufferData(GL_ARRAY_BUFFER, 4 * va.size, va, GL_DYNAMIC_DRAW)
        if ea is not None:
            ea = ea.astype(np.uint32)
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, 4 * ea.size, ea, GL_DYNAMIC_DRAW)
        glDrawElements(ele, len(ea), GL_UNSIGNED_INT, None)

    def layout(self, shader, **attribs):
        glBindVertexArray(self.vao)
        tot_size = sum(attribs.values())
        offset = 0
        for tag, size in attribs.items():
            loc = glGetAttribLocation(shader.program, tag)
            assert loc >= 0
            glVertexAttribPointer(loc, size, GL_FLOAT, GL_FALSE, tot_size * 4, ctypes.c_void_p(offset))
            glEnableVertexAttribArray(loc)
            offset += size * 4

    def release(self):
        self.bind()
        glDeleteVertexArrays(1, GL_VERTEX_ARRAY)
        glDeleteBuffers(1, GL_ARRAY_BUFFER)
        glDeleteBuffers(1, GL_ELEMENT_ARRAY_BUFFER)


class FrameBuffer:

    def __init__(self, w, h):
        self.fbo = glGenFramebuffers(1)
        self.tcb = glGenTextures(1)
        # self.rbo = glGenRenderbuffers(1)
        self.bind()
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, w, h, 0, GL_RGBA, GL_FLOAT, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.tcb, 0)

        if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
            print("[ERROR]", "Frame buffer is not complete!")

        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    def bind(self):
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        glBindTexture(GL_TEXTURE_2D, self.tcb)
        # glBindRenderbuffer(GL_RENDERBUFFER, self.rbo)

    def unbind(self):
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glBindTexture(GL_TEXTURE_2D, 0)
        # glBindRenderbuffer(GL_RENDERBUFFER, 0)

    def release(self):
        self.bind()
        glDeleteBuffers(1, GL_FRAMEBUFFER)
        glDeleteTextures(1, GL_TEXTURE_2D)
        # glDeleteRenderbuffers(1, GL_RENDERBUFFER)
        self.unbind()


def init_glfw(w, h, debug_name=None):
    if not glfw.init():
        raise Exception("[ERROR]", "glfw init failed!")

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)
    if debug_name is not None:
        window = glfw.create_window(w, h, debug_name, None, None)
    else:
        glfw.window_hint(glfw.VISIBLE, GL_FALSE)
        window = glfw.create_window(w, h, "canvas", None, None)
    if not window:
        print("[ERROR]", "glfw init failed!")
        exit(1)
    glfw.make_context_current(window)
    return window


class Rasterizor:

    def __init__(self, w, h, vs, fs, clear=(0, 0, 0,)):
        self.w = w
        self.h = h
        self.window = init_glfw(w, h)
        self.shader = Shader(vs, fs)
        self.buffer = Buffer()
        self.frame = FrameBuffer(w, h)
        self.clear = clear

    def layout(self, **kwargs):
        self.buffer.layout(self.shader, **kwargs)

    def render(self, v_arr, idx):
        self.frame.bind()
        self.buffer.bind()
        glEnable(GL_DEPTH_TEST)
        glViewport(0, 0, self.w, self.h)
        glClearColor(*self.clear, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self.shader.use()
        self.buffer.draw(v_arr, idx)

        data = (GLfloat * (4 * self.w * self.h))(0)
        glReadPixels(0, 0, self.w, self.h, GL_RGBA, GL_FLOAT, data)
        image = np.frombuffer(data, np.float32, 4 * self.w * self.h).reshape([self.h, self.w, 4])
        return np.flip(image, 0)

    def release(self):
        self.shader.release()
        self.buffer.release()
        glfw.terminate()


@contextmanager
def texture_rasterizor(resolution):
    vs = """
         #version 330
         in vec3 position;
         in vec3 color;
         out vec3 newColor;
         void main()
         {
             gl_Position = vec4(position * 2 - 1, 1.0f);
             newColor = color;
         }
         """

    fs = """
         #version 330
         in vec3 newColor;
         out vec4 outColor;
         void main()
         {
             outColor = vec4(newColor, 1.0f);
         }
         """

    rst = Rasterizor(resolution, resolution, vs, fs)
    rst.layout(position=3, color=3)

    def render_texture(uvs, faces, vertex_colors):
        uvw = np.concatenate([uvs, np.zeros_like(uvs[..., :1])], -1)
        v_arr = np.hstack([uvw, vertex_colors]).flatten().astype(np.float32)
        idx = faces.flatten().astype(np.uint32)
        return rst.render(v_arr, idx)

    yield render_texture
    rst.release()


if __name__ == '__main__':
    mesh = trimesh.load("dev/lego_uv3.obj")
    with texture_rasterizor(1024) as tex_render:
        image = tex_render(mesh.visual.uv, mesh.faces, mesh.vertices)
        imageio.imwrite("tmp.exr", image)
