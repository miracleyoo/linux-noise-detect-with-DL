# coding: utf-8
# Author: Zhongyang Zhang

import time
import pickle
import matlab.engine
import sys
import os
from global_val import *
import requests
import json
import numpy as np

# files = {'image01': open('01.jpg', 'rb')}
# user_info = {'name': 'letian', 'data': json.dumps([1, 2, 3, 4.0])}
# output = requests.post("http://127.0.0.1:5000/get-output", data=user_info)#, files=files
# print(output.text)

sys.path.append(os.path.dirname(sys.path[0]))
sys.path.append('./matlab_source_code')
url = "http://localhost:5000/get-output"


class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print('==> [%s]:\t' % self.name, end='')
        print('Elapsed Time: %s (s)' % (time.time() - self.tstart))


def gen_input(net_charge_f):
    net_charge_f = np.array(net_charge_f)[np.newaxis, :]/1e23
    border_cond = np.zeros((1, int(ny1), int(nx1)))
    border_cond[0, :, 0] = Vp
    border_cond[0, :, -1] = Vn
    model_input_f = np.concatenate((net_charge_f, border_cond), axis=0)
    return model_input_f[np.newaxis, :]


def gen_new_input(net_charge_f, last_phi_f=None):
    if last_phi_f is None:
        last_phi_f = np.zeros((1, int(ny1), int(nx1)))
    net_charge_f = np.array(net_charge_f)[np.newaxis, :]/1e23
    last_phi_f = np.array(last_phi_f)[np.newaxis, :]
    border_cond = np.zeros((1, int(ny1), int(nx1)))
    border_cond[0, :, 0] = Vp
    border_cond[0, :, -1] = Vn
    model_input_f = np.concatenate((net_charge_f, border_cond, last_phi_f), axis=0)
    return model_input_f[np.newaxis, :]


eng = matlab.engine.start_matlab()
eng.addpath('./matlab_source_code')
Vp_all = [2.0]#, 1.2, 0.5, 0.8, 1, 1.6, 1.8, 2.25, 2.7]
Vn_all = [0]
USE_DL = True

with Timer('init_core'):
    nx1, ny1 = eng.init_core(matlab.double(Vp_all), matlab.double(Vn_all), nargout=2)

# with Timer('init_dl_core'):
#     opt, net = dl_init()

for a in range(len(Vp_all)):
    for b in range(len(Vn_all)):
        save_prefix = 'Vp=' + str(Vp_all[a] * 100) + 'Vn=' + str(Vn_all[b] * 100) + \
                      'dx=' + str(dx * 1000000000.0) + 'nm'
        if USE_DL:
            save_prefix += '_USE_DL'
        else:
            save_prefix += '_USE_SIM'
        save_name = './source/simulation_res/intermediate_file/' + save_prefix + '.mat'
        simu_res = []

        with Timer('subinit_core'):
            net_charge, Vp, Vn = eng.subinit_core(save_name, float(a + 1), float(b + 1), nargout=3)

        phi = matlab.double(np.zeros((int(ny1), int(nx1))).tolist())

        phi_backup = phi
        if USE_DL:
            with Timer('dl_solver'):
                inputs = {'data': json.dumps(gen_input(net_charge).tolist())}
                phi = np.array(
                    json.loads(requests.post(url, data=inputs).text))
            simu_res.append((gen_new_input(net_charge, last_phi_f=phi_backup), phi))
            phi = matlab.double(phi.squeeze().reshape(41, 9).T.tolist())
            fx, fy = eng.fxy_core(phi, nargout=2)
        else:
            with Timer('pn_poisson_v5'):
                fx, fy, phi = eng.pn_poisson_v5(phi, save_name, nargout=3)
            simu_res.append((gen_new_input(net_charge, last_phi_f=phi_backup), phi))

        for ti in range(0, tsteps):
            with Timer('iteration_core'):
                net_charge = eng.iteration_core(fx, fy, float(ti + 1), save_name, nargout=1)

            phi_backup = phi
            if USE_DL:
                with Timer('dl_solver'):
                    inputs = {'data': json.dumps(gen_input(net_charge).tolist())}
                    phi = np.array(
                        json.loads(requests.post(url, data=inputs).text))
                simu_res.append((gen_new_input(net_charge, last_phi_f=phi_backup), phi))
                phi = matlab.double(phi.squeeze().reshape(41, 9).T.tolist())
                fx, fy = eng.fxy_core(phi, nargout=2)
            else:
                with Timer('pn_poisson_v5'):
                    fx, fy, phi = eng.pn_poisson_v5(phi, save_name, nargout=3)
                simu_res.append((gen_new_input(net_charge, last_phi_f=phi_backup), phi))

            with Timer('statistics_core'):
                eng.statistics_core(float(ti + 1), save_name, nargout=0)

            print('==> progress: %d finished.' % (ti + 1))
        print("==> Simulation Finished(Vn=%f,Vp=%f). %s file saved." % (Vn, Vp, save_name))
        pickle.dump(simu_res, open('./source/simulation_res/train_data/' + save_prefix + '.pkl', 'wb+'))
