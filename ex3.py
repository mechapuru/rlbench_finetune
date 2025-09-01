from OpenGL import GL
from OpenGL import GLU
import glfw

# Initialize GLFW
if not glfw.init():
    raise Exception("GLFW can't be initialized")

# Create hidden window (no actual display needed)
glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
window = glfw.create_window(1, 1, "CheckGL", None, None)
glfw.make_context_current(window)

# Query OpenGL info
vendor = GL.glGetString(GL.GL_VENDOR).decode()
renderer = GL.glGetString(GL.GL_RENDERER).decode()
version = GL.glGetString(GL.GL_VERSION).decode()

print("OpenGL Vendor:", vendor)
print("OpenGL Renderer:", renderer)
print("OpenGL Version:", version)

# Cleanup
glfw.destroy_window(window)
glfw.terminate()
