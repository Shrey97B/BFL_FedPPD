#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse

def args_parser():
	parser = argparse.ArgumentParser()
	
	# data distribution
	parser.add_argument('--rounds', type=int, default=40, help="rounds of training")
	parser.add_argument('--num_users', type=int, default=10, help="number of users")
	parser.add_argument('--num_data', type=int, default=40000, help="number of data distributed to users")
	parser.add_argument('--img_use_frac', type=float, default=1.0, help="Number of images from a shard in a shard based distribution")
	parser.add_argument('--frac', type=float, default=1.0, help="the fraction of clients")

	# multiprocessing
	parser.add_argument('--device_id', type=int, default=2, help="GPU ID to be used")
	parser.add_argument('--num_threads', type=int, default=1, help="number of maximum threads that can be spawned")
	
	#Configuration arguments
	parser.add_argument('--femnist_data_dir', type=str, default='.', help='Location of data file for train and test directory')
	parser.add_argument('--subsample', type=int, default='-1', help='approximate subsample size (only for femnist) for each client if needed, otherwise keep -1')
	parser.add_argument('--config_file', type=str, default=None, help='path of config file containing all the hyperparameters')

	#FedPPD
	parser.add_argument('--teach_lr', type=float, default=0.01, help="v1 Teacher learning rate")
	parser.add_argument('--teach_wd', type=float, default=0.001, help="v1 Teacher Weight Decay")
	parser.add_argument('--teach_sch_gamma', type=float, default=1.0, help="v1 Teacher Scheduler Gamma")
	parser.add_argument('--teach_sch_step', type=int, default=3, help="v1 Teacher Scheduler Step Size")
	parser.add_argument('--stud_lr', type=float, default=0.01, help="v1 Student learning rate")
	parser.add_argument('--stud_wd', type=float, default=0.001, help="v1 Student Weight Decay")
	parser.add_argument('--weight_decay', type=float, default=0.0001, help="weight_decay during  training at server")
	parser.add_argument('--use_SWA', action='store_true', help="use_SWA") 
	parser.add_argument('--use_oracle', action='store_true', help="use_oracle")  
	parser.add_argument('--stud_ent_avg', action='store_true', help="enable it for entropy based aggregation of student model at server")
	parser.add_argument('--distill', action='store_true', help='enable it for distillation at server')
	# client training
	parser.add_argument('--local_ep', type=int, default=10, help="the number of local epochs")
	parser.add_argument('--local_bs', type=int, default=40, help="local batch size")
	
	# server training
	parser.add_argument('--teach_server_lr', type=float, default=0.001, help="v2 Server learning rate - Teacher")
	parser.add_argument('--stud_server_lr', type=float, default=0.001, help="v2 Server learning rate - Student")	
	
	# logging
	parser.add_argument('--log_dir', type=str, default='log', help='model name')
	parser.add_argument('--log_ep', type=int, default=5, help='log_ep')
		
	# dataset
	parser.add_argument('--dataset', type=str, default='cifar10', help="name of dataset")
	parser.add_argument('--iid', action='store_true', help='whether i.i.d or not')
	parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
	parser.add_argument('--num_channels', type=int, default=3, help="number of channels of imges")
	parser.add_argument('--split_method', type=str, default='step', help='split_method, [step, dir]')

	# SWAG & Server
	parser.add_argument('--fedM', action='store_true', help="FedAvgM") 
	parser.add_argument('--teacher_type', type=str, default='SWAG', help='ensemble')
	parser.add_argument('--client_type', type=str, default='real', help='real, g')
	parser.add_argument('--swag_stepsize', type=float, default=1.0, help="swag_stepsize")
	parser.add_argument('--client_stepsize', type=float, default=1.0, help="client_stepsize")
	parser.add_argument('--var_scale', type=float, default=0.1, help="var_scale")
	parser.add_argument('--num_sample_teacher', type=int, default=10, help="number of teachers")
	parser.add_argument('--num_base', type=int, default=20, help="number of teachers")
	parser.add_argument('--use_client', action='store_true', help="use_client")
	parser.add_argument('--use_fake', action='store_true', help="use_fake")
	parser.add_argument('--soft_vote', action='store_true', help='Enable Soft Vote at Server')
	parser.add_argument('--sample_teacher', type=str, default="gaussian", help="use_client")
	parser.add_argument('--loss_type', type=str, default='KL', help='server loss')
	parser.add_argument('--temp', type=float, default=0.5, help="temp")
	parser.add_argument('--mom', type=float, default=0.9, help="teacher momentum")
	parser.add_argument('--server_bs', type=int, default=128, help="server batch size: B")    
	parser.add_argument('--update', type=str, default='dist', help='Aggregation update strategy, [FedAvg, dist]')
	parser.add_argument('--server_ep', type=int, default=20, help="the number of center epochs")
	parser.add_argument('--warmup_ep', type=int, default=-1, help="the number of warmup rounds")

	# model arguments
	parser.add_argument('--model', type=str, default='cnn', help='model name')
	parser.add_argument('--diff_stud_model', type=int, default=0, help="set to 1 to use different student model")

	# other arguments
	parser.add_argument('--gpu', type=int, default=1, help="GPU ID, -1 for CPU")
	parser.add_argument('--verbose', action='store_true', help='verbose print')
	parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
	
	args = parser.parse_args()
	
	if args.update =="FedAvg": args.use_SWA = False
	if args.teacher_type != "SWAG": args.dont_add_fedavg = True
	
	return args
