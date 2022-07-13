# -*- coding: utf-8 -*-
"""
Created on Thu May  5 09:52:11 2022

@author: Bouazzaoui Mohammed
"""


def debug(DEBUG, m):
    ############################################################
    # Function : debug, prints debugger information on terminal
    #
    # Input :  Flag, Message
    # Return :   Message
    ############################################################    
    
    if DEBUG:
        print("\nDEBUG@@@:---", m, "---@@@\n")
