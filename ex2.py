from OpenGL import GL
import OpenGL.EGL as egl
import ctypes

# 1. Get default display
display = egl.eglGetDisplay(egl.EGL_DEFAULT_DISPLAY)
assert display != egl.EGL_NO_DISPLAY, "Failed to get EGL display"

# 2. Initialize EGL
major, minor = egl.EGLint(), egl.EGLint()
ok = egl.eglInitialize(display, major, minor)
assert ok, "eglInitialize failed"
print("EGL version:", major.value, minor.value)

# 3. Choose EGL config
config_attribs = [
    egl.EGL_SURFACE_TYPE, egl.EGL_PBUFFER_BIT,
    egl.EGL_RENDERABLE_TYPE, egl.EGL_OPENGL_BIT,
    egl.EGL_NONE,
]
num_configs = egl.EGLint()
config = egl.EGLConfig()
egl.eglChooseConfig(display, config_attribs, ctypes.byref(config), 1, ctypes.byref(num_configs))
assert num_configs.value > 0, "No matching EGL configs found"

# 4. Create a pbuffer surface (offscreen)
pbuffer_attribs = [
    egl.EGL_WIDTH, 256,
    egl.EGL_HEIGHT, 256,
    egl.EGL_NONE,
]
surface = egl.eglCreatePbufferSurface(display, config, pbuffer_attribs)
assert surface != egl.EGL_NO_SURFACE, "Failed to create EGL surface"

# 5. Bind OpenGL API
egl.eglBindAPI(egl.EGL_OPENGL_API)

# 6. Create an OpenGL context
context = egl.eglCreateContext(display, config, egl.EGL_NO_CONTEXT, None)
assert context != egl.EGL_NO_CONTEXT, "Failed to create EGL context"

# 7. Make it current
egl.eglMakeCurrent(display, surface, surface, context)

# 8. Query OpenGL info
renderer = GL.glGetString(GL.GL_RENDERER).decode()
version = GL.glGetString(GL.GL_VERSION).decode()
vendor = GL.glGetString(GL.GL_VENDOR).decode()

print("OpenGL vendor:", vendor)
print("OpenGL renderer:", renderer)
print("OpenGL version:", version)

# 9. Cleanup
egl.eglDestroySurface(display, surface)
egl.eglDestroyContext(display, context)
egl.eglTerminate(display)
