#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2015, Alex J. Champandard
# Copyright (c) 2015, Vispy Development Team.
# Distributed under the (new) BSD License.

from __future__ import (unicode_literals)

import sys
import datetime
import subprocess

import numpy
import scipy.misc

import vispy
from vispy import gloo
from vispy import app


vertex = """
#version 120

attribute vec2 position;
void main()
{
    gl_Position = vec4(position, 0.0, 1.0);
}
"""

fragment = """
#version 120

uniform vec3      iResolution;           // viewport resolution (in pixels)
uniform float     iGlobalTime;           // shader playback time (in seconds)
uniform vec4      iMouse;                // mouse pixel coords
uniform vec4      iDate;                 // (year, month, day, time in seconds)
uniform float     iSampleRate;           // sound sample rate (i.e., 44100)
uniform sampler2D iChannel0;             // input channel. XX = 2D/Cube
uniform sampler2D iChannel1;             // input channel. XX = 2D/Cube
uniform sampler2D iChannel2;             // input channel. XX = 2D/Cube
uniform sampler2D iChannel3;             // input channel. XX = 2D/Cube
uniform vec3      iChannelResolution[4]; // channel resolution (in pixels)
uniform float     iChannelTime[4];       // channel playback time (in sec)

%s

void main()
{
    mainImage(gl_FragColor, gl_FragCoord.xy);
}
"""


def get_idate():
    now = datetime.datetime.now()
    utcnow = datetime.datetime.utcnow()
    midnight_utc = datetime.datetime.combine(utcnow.date(), datetime.time(0))
    delta = utcnow - midnight_utc
    return (now.year, now.month, now.day, delta.seconds)


def noise(resolution=64, nchannels=1):
    size = (resolution, resolution, nchannels)
    return numpy.random.randint(low=0, high=256, size=size).astype(numpy.uint8)


class RenderingCanvas(app.Canvas):

    def __init__(self, glsl, stdout=None):
        app.Canvas.__init__(self, keys='interactive', size=(960, 540), title='ShaderToy Renderer')
        self.program = gloo.Program(vertex, fragment % glsl)
        self.program["position"] = [(-1, -1), (-1, 1), (1, 1), (-1, -1), (1, 1), (1, -1)]
        self.program['iMouse'] = 0.0, 0.0, 0.0, 0.0
        self.program['iSampleRate'] = 44100.0

        for i in range(4):
            self.program['iChannelTime[%d]' % i] = 0.0
        self.program['iGlobalTime'] = 0.0

        self.activate_zoom()

        self._stdout = stdout
        self._timer = app.Timer('auto', connect=self.on_timer, start=True)
        self.show()

    def set_channel_input(self, img, i=0):
        tex = gloo.Texture2D(img)
        tex.interpolation = 'linear'
        tex.wrapping = 'repeat'
        self.program['iChannel%d' % i] = tex
        self.program['iChannelResolution[%d]' % i] = img.shape

    def on_draw(self, event):
        self.program['iGlobalTime'] += 1.0 / 30.0
        self.program.draw()

        if self._stdout is not None:
            framebuffer = vispy.gloo.util._screenshot((0, 0, self.physical_size[0], self.physical_size[1]))
            scipy.misc.imsave(self._stdout, framebuffer, format='png')

    def on_mouse_click(self, event):
        print('on_mouse_click', event)
        imouse = event.pos + event.pos
        self.program['iMouse'] = imouse

    def on_mouse_move(self, event):
        if event.is_dragging:
            x, y = event.pos
            px, py = event.press_event.pos
            imouse = (x, self.size[1] - y, px, self.size[1] - py)
            self.program['iMouse'] = imouse

    def on_timer(self, event):
        # self.program['iGlobalTime'] = event.elapsed
        self.program['iDate'] = get_idate()
        self.update()

    def on_resize(self, event):
        self.activate_zoom()

    def activate_zoom(self):
        gloo.set_viewport(0, 0, *self.physical_size)
        self.program['iResolution'] = (self.physical_size[0], self.physical_size[1], 0.)


if __name__ == '__main__':
    vispy.set_log_level('WARNING')
    vispy.use(app='glfw')

    outf = 'test.mp4'
    rate = 30
    cmdstring = ('/usr/local/bin/ffmpeg',
                 '-loglevel', 'panic',
                 '-r', '%d' % rate,
                 '-f','image2pipe',
                 '-pix_fmt', 'yuv420p',
                 '-vcodec', 'png',
                 '-i', 'pipe:',
                 '-c:v', 'libx264',
                 outf
                 )
    ffmpeg = subprocess.Popen(cmdstring, stdin=subprocess.PIPE)

    glsl = open('example.glsl', 'r').read()    
    canvas = RenderingCanvas(glsl, stdout=ffmpeg.stdin)
    canvas.set_channel_input(noise(resolution=256, nchannels=2), i=0)
    canvas.set_channel_input(noise(resolution=256, nchannels=2), i=1)

    canvas.show()
    
    try:
        canvas.app.run()
    except KeyboardError:
        pass
        
    ffmpeg.stdin.close()
    ffmpeg.wait()
