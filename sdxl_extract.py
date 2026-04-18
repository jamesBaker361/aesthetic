import os
import argparse
from experiment_helpers.gpu_details import print_details
from experiment_helpers.argprint import print_args


import time

from experiment_helpers.init_helpers import default_parser,repo_api_init

parser=default_parser()
parser.add_argument("--src_dir",type=str,default="laion")
            

def main(args):
    return


    
        


if __name__=='__main__':
    print_details()
    start=time.time()
    args=parser.parse_args()
    print_args(parser)
    print(args)
    main(args)
    end=time.time()
    seconds=end-start
    hours=seconds/(60*60)
    print(f"successful generating:) time elapsed: {seconds} seconds = {hours} hours")
    print("all done!")