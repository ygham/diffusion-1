#!/usr/bin/env python

#from IPython import get_ipython
#ipython = get_ipython()
#ipython.magic("reset -f")
#from tabulate import tabulate # temporary

import os, sys
import numpy as np
import pandas as pd
import seaborn as sns

import random
random.seed(2434523)
from itertools import cycle
import matplotlib.pyplot as plt
plt.style.use('ggplot')

color_cycle = [
'#08088A',
'#8A0808',
'#6A0888',
'#4B8A08',
'#886A08',
'#424242',
'#8A0829',
'#088A68',
'#58FAAC',
'#FAAC58',
'#848484',
'#82FA58',
'#D358F7',
'#FA5858',
'#FA5882',
'#5858FA',
]
color_cycler = cycle(color_cycle)

figsize = (4,4)

class Student():
    def __init__(self, location, rotation):
        self.location, self.rotation = location, rotation
        self.molecule = None # either None, or currently held molecule
        self.outgoing_offer_target = None
        self.incoming_offer_tally = 0 # keep track of incoming offers
        self.leaving_theatre = False

        # note bottom is assumed to be front of theatre
        self.forward = (self.location[0]+1, self.location[1])
        self.right = (self.location[0], self.location[1]-1)
        self.back = (self.location[0]-1, self.location[1])
        self.left = (self.location[0], self.location[1]+1)

    
        self.neighbours = [self.forward, self.right, self.back, self.left]
        if rotation == 'anticlockwise':
            self.neighbours = [self.back, self.right, self.forward, self.left]
        
        self.in_right_column = False
        self.in_left_column = False
        self.in_front_row = False
        self.in_back_row = False
        
        if self.location[0] == n_rows-1: # we're in the front row
            self.in_front_row = True
        if self.location[1] == 0: # right column
            self.in_right_column = True
        if self.location[0] == 0: # back row
            self.in_back_row = True
        if self.location[1] == n_columns-1: # left column
            self.in_left_column = True

        if (self.in_left_column and self.left in self.neighbours): 
            self.neighbours.remove(self.left) # if we're in the left, offering to left is not an option
        if (self.in_right_column and self.right in self.neighbours): 
            self.neighbours.remove(self.right) # if we're in the right, offering to right is not an option
        if (self.in_front_row and self.forward in self.neighbours):
            self.neighbours.remove(self.forward) # if we're in the front, offering to forward is not an option
        if (self.in_back_row and self.back in self.neighbours):
            self.neighbours.remove(self.back) # if we're in the back, offering to back is not an option
        self.neighbours_count = len(self.neighbours)
        self.cycler = cycle(self.neighbours)
        self.cycler.next() 

    def make_offer(self):
        for i in range(self.neighbours_count):
            target = self.cycler.next()
            try:
                if theatre[target].molecule:
                    continue # target student already has a molecule so move onto the next one
                else:
                    self.outgoing_offer_target = target
                    theatre[self.outgoing_offer_target].receive_offer()
                    break # we've made an offer to a target student so stop looking
            except:
                self.leaving_theatre = True # tag as leaving, we'll remove it after offers are tallied


    def receive_offer(self):
        self.incoming_offer_tally += 1 

    def offload_molecule(self):
        if self.leaving_theatre:
            self.molecule = None
            self.outgoing_offer_target = None
        else:
            target_student = theatre[self.outgoing_offer_target]
            if (target_student.incoming_offer_tally < 2): # only offload based on rules of the game
                target_student.receive_molecule(self.molecule)
                self.molecule = None
                self.outgoing_offer_target = None

    def receive_molecule(self, molecule):
        self.molecule = molecule
        self.molecule.position = self.location
        self.molecule.location_history.append(self.location)

    def clear_offer_tally(self):
        self.incoming_offer_tally = 0 # reset the incoming offer tally

class Molecule():
    def __init__(self, identity, initial_position):
        self.identity, self.position = identity, initial_position
        self.location_history = [self.position]
        self.color = color_cycler.next()
    
def fill_theatre():
    # populate the theatre with students
    rotation_options = ['clockwise', 'anticlockwise']
    identity = -1

    for row in range(theatre.shape[0]):
        for column in range(theatre.shape[1]):
            rotation_choice = random.sample(rotation_options, 1) #randomly assign clockwise and anticlockwise rotations
            theatre[row, column] = Student((row, column), rotation=rotation_choice)

    # introduce molecules into the theatre
    initial_molecules = []

    def fill_column(column):
        global identity
        # give each student in the column a molecule
        for row in range(n_rows):
            initial_molecules.append((row,column))

    if right_source:
        fill_column(0)

    if left_source:
        fill_column(n_columns-1)

    if interior_fill:
        for column in interior_fill:
            fill_column(column)

    for location in initial_molecules:
        identity = identity+1
        theatre[location].molecule = Molecule(identity, initial_position=location)


def calc_state():
    state = np.zeros_like(theatre)
    for row in range(theatre.shape[0]):
        for column in range(theatre.shape[1]):
            if theatre[row, column].molecule:
                state[row, column] = 1
            else:
                state[row, column] = 0
    column_occupancy = state.sum(axis=0)
    return state, column_occupancy

def print_info():
    state, column_occupancy = calc_state()
    print state
    #print column_occupancy
    if not state.any():
        print "INFO: everything gone at step {}".format(global_i)
        sys.exit()

def apply_sink(column):
    # before any futher movement, we empty the sink column
    for student in theatre[:,column]: # loop up the column
        if student.molecule:
            student.leaving_theatre = True
            student.offload_molecule()

def apply_source(column):
    global identity
    for student in theatre[:,column]: # loop down the right column
        #if ((not student.molecule) and (student.location[0] % 2)): # only fill up odd rows
        if not student.molecule: # fill up all rows
            identity = identity+1
            theatre[student.location].molecule = Molecule(identity, initial_position=student.location)

def on_clap(figures=False):
    global identity

    # before any further movement, we fill the source columns
    if left_source:
        apply_source(-1) # -1 implies select end column
    
    if right_source:
        apply_source(0)

    if left_sink:
        apply_sink(-1) # -1 implies select end column

    if right_sink:
        apply_sink(0)

    # measure prior to offloading
    state, column_occupancy = calc_state()
    column_occupancy_list.append(column_occupancy)

    for student in theatre.flatten():
        if student.molecule: # the student has a molecule
            student.make_offer()

    # now we see where the molecules move based on all the offer information
    # want to loop over students who currently have molecules

    to_offload = []
    for student in theatre.flatten():
        if (student.molecule and student.outgoing_offer_target):
            to_offload.append(student)

    if figures:
        # generate map and distribution figures
        gen_figures()

    # this step cauase molecules to disappear
    for student in to_offload:
        student.offload_molecule()

    for student in theatre.flatten():
        student.clear_offer_tally()

def gen_figures():
    global global_clap
    global high_res
    map_figure = plt.figure(figsize=figsize)
    for student in theatre.flatten():
        if student.molecule:
            row, column = zip(*student.molecule.location_history)
            color = student.molecule.color
            if global_clap:
                plt.plot(column, row, '-', color=color)
            markersize=4
            if not high_res:
                markersize = 15
            plt.plot(column[-1], row[-1], '-o', markersize=markersize,
                    color=color, mec=color, mfc=color, 
                    zorder=100
                    )
            plt.xlim([-0.5,n_columns-1+0.5])
            if high_res:
                plt.xlim([-0.5,n_columns-1+1])
            plt.ylim([-0.5,n_rows-1+0.5])
            if high_res:
                plt.ylim([-0.5,n_rows-1+1])

            plt.gca().invert_yaxis()
            plt.xlabel('column')
            plt.ylabel('row')
            plt.gca().set_aspect('equal', adjustable='box')
            if not high_res:
                plt.xticks(range(n_columns))
                plt.yticks(range(n_rows))
    name = 'images/{}_map_{:05d}.png'.format(tag,global_clap)
    if high_res:
        name = 'images/{}_map_{:05d}.png'.format(tag,global_clap)
    plt.savefig(name, bbox_inches='tight')
    #plt.show()
    plt.close(map_figure)

    clean = []
    occupancy = np.vstack(column_occupancy_list)
    for claps, counts in enumerate(occupancy):
        for column, count in enumerate(counts):
            clean.append([claps, column, count])
    clean = np.vstack(clean)
    df = pd.DataFrame(clean, columns=['claps', 'column', 'count'])


    distribution = plt.figure(figsize=figsize)
    ax = sns.boxplot(x="column", y="count", data=df, sym='')
    if not high_res:
        ax = sns.stripplot(x="column", y="count", data=df, size=4, jitter=True, edgecolor="gray")
    ax.set_ylim([-0.5,n_rows+0.5])
    if high_res:
        ax.set_ylim([-0.5,n_rows+2])


    if high_res:
        ax.set_xticks(range(0,n_columns+1,10))
        ax.set_xticklabels(range(0,n_columns+1,10))
        ax.set_yticks(range(0,n_rows+1,10))
    else:   
        ax.set_yticks(range(n_rows+1))

    name = 'images/{}_distribution_{:05d}.png'.format(tag,global_clap)
    if high_res:
        name = 'images/{}_distribution_{:05d}.png'.format(tag,global_clap)
    plt.savefig(name, bbox_inches='tight')
    #plt.show()
    plt.close(distribution) 


def run_case(tag):
    global global_clap
    global identity
    print "INFO: running case ", tag
    fill_theatre()
    for global_clap in range(2001):
        print global_clap
        figure_output = False
        if global_clap in [0,1,2,3,5,10,20,30,40,50,100,200,300,400,500,1000,2000]:
            figure_output = True
        on_clap(figures=figure_output)


# check if images exists, we need it for storing figures
if not os.path.isdir('images'):
    os.mkdirs('images')

# not ideal, but store these as global variables
# next time I do something like this, make a run_case() class and store required theatre arrays within the instance
identity=0
global_clap=0

# run the various cases

tag ='source_right_sink_left'
n_rows = 10
n_columns = 10
high_res = False
interior_fill = []
right_source = True
left_source = False
right_sink = False
left_sink = True
# reset theatre
theatre = np.zeros((n_rows,n_columns), dtype=object)
column_occupancy_list = []
run_case(tag)

#tag ='source_right_sink_left_high_res'
#n_rows = 50
#n_columns = 50
#high_res = True
#interior_fill = []
#right_source = True
#left_source = False
#right_sink = False
#left_sink = True
## reset theatre
#theatre = np.zeros((n_rows,n_columns), dtype=object)
#column_occupancy_list = []
#run_case(tag)

#tag ='source_right_source_left'
#n_rows = 10
#n_columns = 10
#high_res = False
#interior_fill = []
#right_source = True
#left_source = True
#right_sink = False
#left_sink = False
#theatre = np.zeros((n_rows,n_columns), dtype=object)
#column_occupancy_list = []
#run_case(tag)

#tag ='source_right_source_left_high_res'
#n_rows = 50
#n_columns = 50
#high_res = True
#interior_fill = []
#right_source = True
#left_source = True
#right_sink = False
#left_sink = False
#theatre = np.zeros((n_rows,n_columns), dtype=object)
#column_occupancy_list = []
#run_case(tag)

#tag ='interior_fill'
#n_rows = 10
#n_columns = 10
#high_res = False
#interior_fill = [4,5]
#right_source = False
#left_source = False
#right_sink = False
#left_sink = False
#theatre = np.zeros((n_rows,n_columns), dtype=object)
#column_occupancy_list = []
#run_case(tag)

#tag ='interior_fill_high_res'
#n_rows = 50
#n_columns = 50
#high_res = True
#interior_fill = [23,24,25]
#right_source = False
#left_source = False
#right_sink = False
#left_sink = False
#theatre = np.zeros((n_rows,n_columns), dtype=object)
#column_occupancy_list = []
#run_case(tag)
