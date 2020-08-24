#!/usr/bin/env python
'''
    AnimateTools
    Library which has a simple and extendable way to animate
    agent-based simulation of active-matter (swarms, colloids, flocks....).
    The animation can be saved as movies, gifs, images of can just be displayed.
    v0.1, 24.8.2020

    (C) 2020 Pascal Klamser

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''
from time import sleep
from scipy import spatial
import numpy as np
import matplotlib
if __name__ == '__main__':
    matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import animation
from functools import partial
import os


def setDefault(x, val):
    if x is None:
        x = val
    return x


def voronoi_Lines(vor):
    """
    Plot the given Voronoi diagram in 2-D

    Parameters
    ----------
    vor : scipy.spatial.Voronoi instance
        Diagram to plot

    Returns
    -------
    finite_segments
    infinite_segments
    """
    if vor.points.shape[1] != 2:
        raise ValueError("Voronoi diagram is not 2-D")

    center = vor.points.mean(axis=0)
    ptp_bound = vor.points.ptp(axis=0)

    finite_segments = []
    infinite_segments = []
    for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
        simplex = np.asarray(simplex)
        if np.all(simplex >= 0):
            finite_segments.append(vor.vertices[simplex])
        else:
            i = simplex[simplex >= 0][0]  # finite end Voronoi vertex

            t = vor.points[pointidx[1]] - vor.points[pointidx[0]]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[pointidx].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[i] + direction * ptp_bound.max()

            infinite_segments.append([vor.vertices[i], far_point])
    return finite_segments, infinite_segments

class voro_lines:
    def __init__(self, preys, axs, preds=None, t0pred=None):
        '''
        INPUT:
            preys datCollector object
                position and velocity data of prey
            axs
            preys datCollector object
                position and velocity data of predators
            t0p int
                time-step at which predator appears
        '''
        self.t0pred = t0pred
        self.pos = preys.pos
        self.posExtra = None
        if preds is not None:
            self.posExtra = preds.pos
        posNow = self.pos[0]
        if preds is not None and t0pred == 0:
            posNow = np.vstack((posNow, self.posExtra[0]))
        vor = spatial.Voronoi(posNow)
        lines_normal, lines_inf = voronoi_Lines(vor)
        self.voro_lines = axs.add_collection(LineCollection(lines_normal,
                                                            colors='b',
                                                            alpha=0.3, lw=0.9))


    def update(self, s):
        posNow = self.pos[s]
        if self.posExtra is not None and s >= self.t0pred:
            posNow = np.vstack((posNow, self.posExtra[s-self.t0pred]))
        posNow = posNow[~np.isnan(posNow)].reshape(-1, 2)
        vor = spatial.Voronoi(posNow)
        lines_normal, lines_inf = voronoi_Lines(vor)
        self.voro_lines.set_segments(lines_normal)
        return self.voro_lines


class velocityArrows:
    def __init__(self, dat, ax, color=None, delay=None):
        '''
        INPUT:
            pos.shape (N, 2)
            vel.shape (N, 2)
        '''
        if color is None:
            color = 'r'
        if delay is None:
            delay = 0 
        self.pos = dat.pos
        self.vel = dat.vel
        self.delay = delay
        self.createFunc = partial(ax.quiver, color=color,
                            angles='xy', scale_units='xy', scale=1)
    def create(self):
        self.Q = self.createFunc(self.pos[0].T[0], self.pos[0].T[1],
                                 self.vel[0].T[0], self.vel[0].T[1])

    def update(self, s):
        s -= self.delay
        if s == 0:
            self.create()
        elif s > 0:
            self.Q.set_offsets(self.pos[s])
            self.Q.set_UVC(self.vel[s].T[0], self.vel[s].T[1])
        if s >= 0:
            return self.Q


def h5ReadDset(h5file, dname):
    data = h5file.get(dname)
    np_data = np.array(data)
    print(np_data)
    if gname in list(h5file.keys()):
        data2 = h5file.get(gname + '/datadouble')
        np_data2 = np.array(data2)
        print(np_data2)


class datCollector:
    def __init__(self, dat):
        if dat is None:
            self.pos = None
            self.vel = None
        else:
            dat = dat.T
            self.pos = dat[:2].T
            self.vel = dat[2:4].T
            if len(dat) > 4:
                self.fitness = dat[4].T
                self.force = dat[5:].T
            dead = np.where( np.sum(self.pos, axis=-1) == 0 )
            self.pos[dead] = np.nan
            self.vel[dead] = np.nan


class headAndTail:
    def __init__(self, data, size, colors, ax, tail_length,
                 cmap=None, scatter=None, marker=None, delay=None):
        '''
        INPUT:
            data.shape(Time, N, N_coord)
                data containing for "Time" time points the position for "N"
                agents.
            size float
                radius of marker size
            colors.shape(N)
                array or list containing for each particle a colorcode (float) 
            ax matplotlib.axes.AxesSubplot object
                Subplot where the data is plotted
            cmap matplotlib.Colormap
                e.g.: cmap = plt.get_cmap('Reds')
            scatter boolean
                if true scatter is used instead of plot
        OUTPUT:
            heads
            tails
        '''
        if delay is None:
            delay = 0
        self.pos = data.pos
        self.tail_length = tail_length
        self.scatter = scatter
        self.delay = delay
        self.ax = ax
        self.sizeRaw = size
        if marker is None:
            marker = 'o'
        if self.scatter is None:
            self.scatter = True
        if self.scatter:
            self.createFunc = partial(ax.scatter, marker=marker, c=colors, cmap=cmap)
        else:
            self.createFunc = partial(ax.plot, marker=marker, c=colors, linestyle='none')


    def sizeAdjust(self):
        trans = self.ax.transData
        pixels = trans.transform([(0, 1), (1, 0)]) - trans.transform((0, 0))
        xpix = pixels[1].max()
        self.size = self.sizeRaw * xpix / 77. * 100
        if self.scatter:
            self.size = self.size ** 2


    def create(self):
        posInit = self.pos[0]   # use Time=1 for initialization
        if self.scatter:
            self.heads = self.createFunc(posInit.T[0], posInit.T[1], s=self.size)
            self.tails = self.createFunc(posInit.T[0], posInit.T[1],
                                         s=self.size/4, alpha=0.5)
        else:
            self.heads = self.createFunc(posInit.T[0], posInit.T[1], ms=self.size)[0]
            self.tails = self.createFunc(posInit.T[0], posInit.T[1],
                                         ms=self.size/2, alpha=0.5)[0]


    def update(self, s):
        self.sizeAdjust()
        s = s - self.delay
        if s == 0:
            self.create()
        elif s > 0:
            endtail = s - self.tail_length
            if endtail < 0:
                endtail = 0
            if self.scatter:
                tail = self.pos[endtail:s].reshape(-1, 2)
                self.heads.set_offsets(self.pos[s])
                self.tails.set_offsets(tail)
                self.heads.set_sizes(len(self.pos[s]) * [self.size])
                self.tails.set_sizes(len(tail) * [self.size / 4])
            else:
                self.heads.set_data(self.pos[s].T[0], self.pos[s].T[1])
                self.tails.set_data(self.pos[endtail:s].T[0],
                                    self.pos[endtail:s].T[1])
                self.heads.set_markersize(self.size)
                self.tails.set_markersize(self.size / 2)
        if s >= 0:
            return self.heads, self.tails


class Limits4Pos:
    '''
    adjust the limits of the axis to the positions of the agents
    -> always see all agents and not more
    '''
    def __init__(self, positions, ax, delay=None):
        '''
        Assumes that the start-positions can be different
        but all end synchrounously 
        INPUT:
            positions [posA, posB, ....]
                list of arrays containing positions, i.e.
                    posA.shape = [Time, Nagents, 2] OR [Time, 2]
        '''
        if delay is None:
            delay = 0
        self.delay = delay
        self.ax = ax
        times = [len(pos) for pos in positions]
        limits = np.zeros((max(times), 4)) * np.nan
        for i, limitFunc in enumerate([np.nanmin, np.nanmax]):
            for j in range(2): # x or y
                for k, pos in enumerate(positions):
                    if len(pos.shape) < 3:
                        pos = np.expand_dims(pos, axis=1)
                    limitGlob = limits[:times[k], i*2+j]
                    limitHere = limitFunc(pos[::-1, :, j], axis=1)
                    compareLimits = np.vstack((limitHere, limitGlob))
                    limitGlob[:] = limitFunc(compareLimits, axis=0)
        limits = limits[::-1]
        extra = 1
        self.xMin = limits[:, 0] - extra
        self.yMin = limits[:, 1] - extra
        self.xMax = limits[:, 2] + extra
        self.yMax = limits[:, 3] + extra

    def update(self, s):
        s = s - self.delay
        if s >= 0:
            self.ax.set_xlim(self.xMin[s], self.xMax[s])
            self.ax.set_ylim(self.yMin[s], self.yMax[s])
        return self.ax


class LimitsByCOM:
    def __init__(self, pos, ax, centerSize):
        '''
        INPUT:
            pos.shape [T, N, 2]
        '''
        self.ax = ax
        self.cs = centerSize
        if len(pos.shape) < 3:
            pos = np.expand_dims(pos, axis=1)
        self.com = np.nanmean(pos, axis=1)

    def update(self, s):
        self.ax.set_xlim(self.com[s, 0] - 0.5 * self.cs,
                         self.com[s, 0] + 0.5 * self.cs)
        self.ax.set_ylim(self.com[s, 1] - 0.5 * self.cs,
                         self.com[s, 1] + 0.5 * self.cs)
        return self.ax


class ForceLines:
    def __init__(self, pos, force, ax, magnify=None):
        '''
        INPUT:
            pos.shape [T, N, 2]
            force.shape [T, N, 2]
        '''
        if magnify is None:
            magnify = 0.25
        assert len(pos.shape) == 3, 'wrong shape for ForceLines' 
        self.N = pos.shape[1]
        self.magni = magnify
        self.pos = pos
        self.force = force 
        line = self.posAndForce2Line(0)
        self.forceLines = ax.plot(line[0], line[1])[0]

    def posAndForce2Line(self, s):
        line = []
        for i in range(2):
            line += [np.array(list(zip(self.pos[s, :, i],
                                       self.pos[s, :, i] + self.magni * self.force[s, :, i],
                                       self.N * [None]))).flatten()]
        return line

    def update(self, s):
        line = self.posAndForce2Line(s)
        self.forceLines.set_data(line[0], line[1])
        return self.forceLines

class lineCOM:
    def __init__(self, pos, color, ax):
        self.com = np.nanmean(pos, axis=1)
        self.lineCOM = ax.plot(self.com[0 ,0], self.com[0, 1], c=color)[0]

    def update(self, s):
        self.lineCOM.set_data(self.com[:s ,0], self.com[:s, 1])
        return self.lineCOM

class lineFadingCOM:
    def __init__(self, pos, ax, lineLength, alpha=None):
        if alpha is None:
            alpha = 0.7
        self.N = lineLength
        self.com = np.nanmean(pos, axis=1)
        points = np.array(self.N*[self.com[0]]).reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        norm = plt.Normalize(0, self.N-1)
        lc = LineCollection(segments, cmap='Reds', norm=norm, alpha=alpha)
        lc.set_array(np.arange(self.N))
        self.lc = ax.add_collection(lc)

    def update(self, s):
        if s >= self.N:
            points = self.com[s-self.N+1:s+1].reshape(-1, 1, 2)
        else:
            points = np.vstack(( (self.N-s) * [self.com[0]], 
                                 self.com[:s+1])).reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        self.lc.set_segments(segments)
        return self.lc


class circleCOM:
    def __init__(self, pos, radius, color, ax):
        self.com = np.nanmean(pos, axis=1)
        circle = plt.Circle((self.com[0 ,0], self.com[0, 1]), radius,
                            color=color, fill=False)
        self.circCom = ax.add_artist(circle)

    def update(self, s):
        self.circCom.set_center((self.com[s ,0], self.com[s, 1]))
        return self.circCom


class circlesCOM:
    def __init__(self, pos, radius, color, ax, Ncircs):
        self.com = np.nanmean(pos, axis=1)
        self.circCom = []
        radiis = np.linspace(radius, radius/2, Ncircs)
        for rad in radiis:
            circle = plt.Circle((self.com[0 ,0], self.com[0, 1]), rad,
                                color=color, fill=False)
            self.circCom.append(ax.add_artist(circle))


    def update(self, s):
        for i, circ in enumerate(self.circCom):
            z = s-i
            if z > 0:
                circ.set_center((self.com[z ,0], self.com[z, 1]))
        return tuple(self.circCom)


class taskCollector:
    '''
    collects tasks (=objects with update(s) method returning artist objects)
    '''
    def __init__(self):
        self.taskList = []

    def append(self, task):
        self.taskList.append(task)

    def update(self, s):
        '''
        returns all output summarized
        '''
        artistObjs = ()
        for task in self.taskList:
            artist = task.update(s)
            if type(artist) != tuple:
                artist = tuple([artist])
            # print(type(artistObjs), type(artist), task.__class__)
            artistObjs += artist 
        return artistObjs


def UpdateViaDraw(fig, tasks, tmin, tmax, fps=None, dpi=None,
                  mode=None, name=None):
    fps = setDefault(fps, 15)
    dpi = setDefault(dpi, 300)
    mode = setDefault(mode, 'normal')
    name = setDefault(name, 'Animation')
    maxFps = 20 # this is system specific
    interval = 1/fps - 1/maxFps
    for s in range(tmin, tmax):
        _ = tasks.update(s)
        if(mode != 'movie'):
            fig.canvas.draw()
            if interval > 0:
                sleep(interval)
        if(mode != 'normal'):
            fig.savefig('f%06d.jpg' % s, dpi=dpi)
    if (mode == 'movie'):
        com0 = 'mencoder mf://f*.jpg -mf fps={}:type=jpg'.format(fps)
        com1 = ' -vf scale=-10:-1 -ovc x264 -x264encopts'
        com2 = ' bitrate=2400 -o {}.avi'.format(moviename)
        os.system(com0+com1+com2)
        os.system("rm f*.jpg")
    elif(mode == 'gif'):
        print('gif-representation not implemented')


def UpdateViaAnimation(fig, tasks, tmin, tmax, fps=None, dpi=None,
                       mode=None, name=None, repeat=None):
    fps = setDefault(fps, 15)
    dpi = setDefault(dpi, 300)
    mode = setDefault(mode, 'normal')
    name = setDefault(name, 'Animation')
    name = setDefault(repeat, True)
    interval = 1000*(1/fps)
    anim = animation.FuncAnimation(fig, tasks.update, interval=interval,
                                   frames=range(tmin-1, tmax), repeat=repeat)
    if mode == 'movie':
        anim.save(name + '.mp4', writer='ffmpeg', dpi=dpi)
    elif mode == 'gif':
        anim.save(name + '.gif', writer='imagemagick', dpi=dpi)
    else:
        plt.show()
