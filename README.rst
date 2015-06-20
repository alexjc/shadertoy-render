shadertoy-render
================

A simple Python script that uses ``ffmpeg`` as a subprocess to render ShaderToy scripts into video files.  After grabbing the file from the repository, you can run it as follows:

	> python shadertoy-render.py example.glsl example.mp4

Dependencies include `numpy`, `scipy` and `vispy`.  You can install them with Python as follows:

    > pip install numpy scipy vispy

The output is a MP4 file with default encoding settings, which you can upload to YouTube for example.  See the source code for details!

.. image:: docs/example.jpg
	:target: https://youtu.be/GAauIQFHaZs

1. `Source ShaderToy <https://www.shadertoy.com/view/4sB3D1>`_ script by Inigo Quilez.

2. `Rendered Video <https://youtu.be/GAauIQFHaZs>`_ at 1080p uploaded to YouTube.

Feedback is welcome; just submit a ticket or follow `@alexjc <https://twitter.com/alexjc>`_ on Twitter.