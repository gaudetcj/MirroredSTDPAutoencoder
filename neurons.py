# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 21:45:27 2017

@author: Chase
"""

import numpy as np
from scipy.integrate import ode


class Neuron():
    def __init__(self, Vthresh, Vrest, Vex, Vinh, Vadap, Tmemb):
        self.Vthresh = Vthresh
        self.Vrest = Vrest
        self.Vex = Vex
        self.Vinh = Vinh
        self.Vadap = Vadap
        
        self.t0 = 0
        self.Vinit = np.random.rand(1)*Vthresh
        self._set_up_ode()
        
    def _function(self, t, V, args):
        E, Eonoff, I, Iadap = args
        first = (self.Vrest - V)
        second = (self.Vex - V) * (E + Eonoff)
        third = (self.Vinh - V) * I
        fourth = (self.Vadap - V) * Iadap
        V = (first + second + third + fourth) / self.Tmemb
        return V
        
    def _set_up_ode(self):
        self.r = ode(self._function).set_integrator('dop853', method='bdf')
        self.r.set_initial_value(self.Vinit, self.t0)
    
    def update(self, args, dt):
        self.r.set_f_params(args)
        self.r.integrate(self.r.t+dt)
        if self.r.y[0] >= self.Vthresh:
            t0 = self.r.t
            self.r = ode(self._function).set_integrator('dop853', method='bdf')
            self.r.set_initial_value(self.Vrest, t0)
            self.spiking = 1
        else:
            self.spiking = 0
        return self.r.y[0], self.spiking