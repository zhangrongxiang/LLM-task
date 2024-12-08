import argparse

def parse_args():
    parser=argparse.ArgumentParser(description='configuration')
    parser.add_argument('--layer',type=str, choices=['low','middle','high'])
    parser.add_argument('--location',type=str)
    parser.add_argument('--col',type=int)
    parser.add_argument("--index",type=int,default=0)
    args=parser.parse_args()

    return  args