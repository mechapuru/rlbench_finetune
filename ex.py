from OpenGL import GL
import OpenGL.EGL as egl

display = egl.eglGetDisplay(egl.EGL_DEFAULT_DISPLAY)
egl.eglInitialize(display, None, None)

major, minor = egl.EGLint(), egl.EGLint()
egl.eglInitialize(display, major, minor)
print("EGL version:", major.value, minor.value)
