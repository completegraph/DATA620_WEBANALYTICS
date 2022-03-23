#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 12:26:38 2022

@author: alexanderng
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx
import networkx.algorithms.bipartite as bipartite
import json
import os
import re
import matplotlib.pyplot as plt
import operator

print("Run this script in the Project 2 working directory where data files have been downloaded.\n")



"""
Load the senator data files from json or csv to dataframe
"""
with open("S113_rollcalls.json", 'r') as f:
    rollcall_data = json.load(f)

raw_rollcall = pd.DataFrame(rollcall_data)

stage_nomination_rollcalls = raw_rollcall[ raw_rollcall["bill_number"].notnull() ]

stage_nomination_rollcalls = stage_nomination_rollcalls[ stage_nomination_rollcalls["bill_number"].str.startswith('PN') ]

# Select only useful columns
nomination_rollcalls = stage_nomination_rollcalls[['rollnumber', 'date', 'yea_count', 'nay_count' ,
                                                   'nominate_mid_1' , 'bill_number' , 'vote_result' , 'vote_desc' ,
                                                   'vote_question' , 'clausen_codes' , 'peltzman_codes', 'issue_codes'
                                                   ]]

# Extract the set of nominees based solely on the vote question.
# The name is always the first string in Vote Desc before , of

vote_questions = nomination_rollcalls[ nomination_rollcalls['vote_question'] == 'On the Nomination' ].copy().reset_index(drop=True)

df_nominations = vote_questions['vote_desc'].apply(str.strip).str.split(', of', 1, expand = True).rename(
     columns = {0: 'Name', 1: 'StatePosition'})

df_nominations['StatePosition'] = df_nominations['StatePosition'].apply(str.strip)

state_position = df_nominations['StatePosition'].apply( lambda x: re.split(', to be |, for the rank of ',x ))

nomination_state_pos = pd.DataFrame( state_position.tolist(), columns = ['State', 'Position'] )

issue_codes = vote_questions["issue_codes"].apply(pd.Series).rename( columns = {0: 'issue_code1', 1: 'issue_code2'} )
clausen_codes = vote_questions["clausen_codes"].apply(pd.Series).rename( columns = {0: 'clausen_code1'} )

nominations = pd.concat( [ df_nominations, nomination_state_pos, issue_codes["issue_code1"], clausen_codes, 
                vote_questions[['rollnumber', 'date', 'yea_count', 'nay_count', 'nominate_mid_1', 'bill_number', 'vote_result']]], axis = 1)

#
# This function allows us to convert each nomination to its branch
# based on the department associated with the position.
#
def translate_dept(row):
    
    iss1 = row['issue_code1']
    pos = row['Position']
    
    if iss1 in [ 'Judiciary',  'Banking and Finance' ,  'Energy' ]:
        return iss1
    if operator.contains(pos, "Internal Revenue") or operator.contains(pos, "Bank"):
        return "Banking and Finance"
    
    if operator.contains( pos, "Ambassador") or operator.contains(pos, "Secretary of State"):
        return "State"
    
    if pos in ['Secretary of Defense', 
               'Secretary of the Air Force' 
               ] or operator.contains(pos, "Central Intelligence"):
        return "Defense"
    if operator.contains(pos, "Office of Management and Budget") or \
       operator.contains(pos, "Personnel Management" ):
       return "Executive"
    
    if operator.contains(pos, "Homeland Security"):
        return "Homeland Security"
    if operator.contains(pos, "National Labor Relations") or \
        operator.contains(pos, "Privacy and Civil Liberties") or \
        operator.contains(pos, "Environmental Protection") or \
        operator.contains(pos, "Federal Trade Commission") or \
        operator.contains(pos, "Equal Employment Opportunity Commission") or \
        operator.contains(pos, "Nuclear Regulatory") or \
        operator.contains(pos, "Tennessee Valley") or \
        operator.contains(pos, "Consumer Product Safety"):
        return "Regulatory"
    
    if operator.contains(pos, "Secretary of the Interior") or \
        operator.contains(pos, "Land Management") or \
        operator.contains(pos, "Commerce") or \
        operator.contains(pos, "Transportation") or \
        operator.contains(pos, "of Labor"):
        return "Domestic, Commerce"
    if operator.contains(pos, "Health and Human Services") or \
        operator.contains(pos, "Social Security") or \
       operator.contains(pos, "Housing and Urban Development") or \
       operator.contains(pos, "Veterans") or \
         operator.contains(pos, "Public Health") or \
       operator.contains(pos, "Medicare"):
        return "Human Services"
    if operator.contains(pos, "Federal Bureau of Investigation") or \
       operator.contains(pos, "Attorney General") or \
       operator.contains(pos, "Alcohol, Tobacco, Firearms"):
        return   "Justice"
    
    return 'Other'

nominations["dept"] = nominations.apply(lambda row: translate_dept(row), axis = 1)

# Keep the important columns
nominations = nominations[["Name", "State", "dept", "Position", "clausen_code1", "rollnumber", "date", "yea_count", 
                           "nay_count", "nominate_mid_1", "bill_number", "vote_result"]]


# for debugging only
# All nominations should have a mapped dept field.
# --------------------------------------------------
B = nominations[nominations["dept"]=="Other"][["Name", "dept", "Position", "rollnumber", "clausen_code1", "bill_number"]]

# Load the Senate Members

members_raw = pd.read_csv("S113_members.csv")

#
#  Need to drop Obama.  The president is in the file.
#
df_members = members_raw[ members_raw["chamber"] == "Senate" ]

def convert_party_code(row):
    
    party_code = row["party_code"]
    
    if party_code == 100:
        return "Democrat"
    if party_code == 200:
        return "Republican"
    if party_code == 328:
        return "Independent"
    return "Unknown"

df_members["party"] = df_members.apply( lambda row:  convert_party_code(row), axis = 1 )

def convert_bioname(row):
    
    name = row["bioname"]
    state = row["state_abbrev"]
    party_code = row["party_code"]
    
    X = name.split(", ")
    
    last_name_init =  X[0] 

    if party_code == 200:
        short_code = "R"
    else:
        short_code =  "D"

    return short_code + "-" + state + "-" + last_name_init

df_members["short_name"] = df_members.apply(lambda row: convert_bioname(row), axis = 1)

# Use this dataframe
# --------------------
members = df_members[["icpsr", "state_abbrev", "party", "occupancy", "bioname", "short_name",  "born", "nominate_dim1", "state_icpsr"]]  

members.reset_index(inplace=True)  # Zero out the index

## Finally, let's load the votes for each nomination
# Need to drop the votes of the president.
votes_raw = pd.read_csv("S113_votes.csv")

head_votes = votes_raw.head()  # for testing

head_nominations = nominations.head()  # for testing

# Join with nominations on the raw votes.  But this will include the President's vote which needs to be omitted.
#
join_votes_nominations = pd.merge( nominations, votes_raw , how = 'inner', left_on = "rollnumber", right_on = "rollnumber")

head_join_votes_nominations = join_votes_nominations.head()  # for testing

join_all = pd.merge( join_votes_nominations, members, how = "inner", left_on = "icpsr", right_on = "icpsr")

def convert_castcode(row):
    
    cast_code = row["cast_code"]

    if cast_code == 1:
        return "Yes"
    if cast_code == 6:
        return "No"
    return "Present"

join_all["cast_value"] = join_all.apply( lambda row :  convert_castcode(row), axis = 1 )

head_join_all = join_all.head() # Test

#
# Contains the nominee, roll call result, senator vote and party.
# Useful for building the network graph.
# -------------------------------------------------------------------
join_select = join_all[["Name", "dept", "Position", "clausen_code1", "rollnumber", "date", "yea_count", "nay_count",
                        "bill_number", "icpsr", "cast_value", "state_abbrev", "party", "short_name", "bioname" ]]

# Construct a graph:
#   Senator nodes are identified by icpsr with 
#         attributes:  short_name, party, bioname, state_abbrev
#   
#   Nomination-cast nodes are keyed by <bill_number>_Y and <bill_number>_N
#         with attributes:
#              dept,  Name, Position, rollnumber, yea_count, nay_count
#
#   Edges have no attributes.
# -----------------------------------------------------------------                       
G = nx.Graph()
for r, d in join_select.iterrows():
    
    senator_id = d["icpsr"]
    cast_value = d["cast_value"]
    
    G.add_node( senator_id, bipartite = "Senator" ,
                 short_name = d["short_name"] ,
                 bioname = d["bioname"],
                 party = d["party"] ,
                 state_abbrev = d["state_abbrev"]
               )
    
    nomination_y = d["bill_number"] + "_y"
    nomination_n = d["bill_number"] + "_n"
    
    G.add_node( nomination_y, bipartite = "Nomination", 
                cast_value = cast_value ,
                dept = d["dept"] ,
                name = d["Name"] ,
                position = d["Position"] ,
                rollnumber = d["rollnumber"] ,
                yea_count = d["yea_count"] ,
                nay_count = d["nay_count"] )
    
    G.add_node( nomination_n, bipartite = "Nomination", 
                cast_value = cast_value,
                dept = d["dept"] ,
                name = d["Name"] ,
                position = d["Position"] ,
                rollnumber = d["rollnumber"] ,
                yea_count = d["yea_count"] ,
                nay_count = d["nay_count"] )
    
    if cast_value == "Yes":
        G.add_edge( senator_id, nomination_y)
    if cast_value == "No":
        G.add_edge( senator_id, nomination_n)
                
print(len(G.edges()), len(G.nodes()))

nx.write_graphml( G, "SenateNominations.graphml" )



# Construct a weighted graph
#   Senator nodes are identified by icpsr with 
#         attributes:  short_name, party, bioname, state_abbrev
#   
#   Nomination-cast nodes are keyed by <bill_number>
#         with attributes:
#              dept,  Name, Position, rollnumber, yea_count, nay_count
#
#   Edges have attributes:  Yea or Nay
#
# -----------------------------------------------------------------                       
G_weighted = nx.Graph()
for r, d in join_select.iterrows():
    
    senator_id = d["icpsr"]
    cast_value = d["cast_value"]
    
    G_weighted.add_node( senator_id, bipartite = "Senator" ,
                 short_name = d["short_name"] ,
                 bioname = d["bioname"],
                 party = d["party"] ,
                 state_abbrev = d["state_abbrev"]
               )
    
    nomination = d["bill_number"]
    
    G_weighted.add_node( nomination, bipartite = "Nomination", 
                cast_value = cast_value ,
                dept = d["dept"] ,
                name = d["Name"] ,
                position = d["Position"] ,
                rollnumber = d["rollnumber"] ,
                yea_count = d["yea_count"] ,
                nay_count = d["nay_count"] )
    
    
    if cast_value == "Yes":
        G_weighted.add_edge( senator_id, nomination, cast_value = "Yes" )
    if cast_value == "No":
        G_weighted.add_edge( senator_id, nomination, cast_value = "No" )
                
print(len(G_weighted.edges()), len(G_weighted.nodes()))

nx.write_graphml( G_weighted , "SenateNominations_Weighted.graphml" )


#  Build a biadjacency matrix using the M senators and 2 * N nominations
#  Assign an explicit senator order ex ante to a list
#  Assign an explicit yes-nomination and no-nomination list.  2*N
#  The entries are 1 for vote of Yea or Nay in appropriate column.
#  Abstentions, present, no voting are treated as 0. 
# --------------------------------------------------------------------------
s_rollnumber =  nominations["rollnumber"]  # length is number of columns N  
s_icpsr = members["icpsr"]   # length is number of rows M

M = len(s_icpsr)  # rows
N = len(s_rollnumber)  # columns

m_bia = np.zeros( shape=( M, 2*N ) , dtype = np.int8)  # for senator covoting


for r, d in join_select.iterrows():
    
    row_i = s_icpsr[ s_icpsr == d["icpsr"]  ].index[0]
    
    cast_value = d["cast_value"]
    
    rollnumber = d["rollnumber"]
    
    rollnumber_index = s_rollnumber[ s_rollnumber == rollnumber ].index[0]
    
    if cast_value == "Yes":
        col_j = rollnumber_index
    if cast_value == "No":
        col_j = N + rollnumber_index
    
    m_bia[ row_i, col_j ] = 1


np.savetxt("Nomination_Biadjacency_Matrix.csv", m_bia, delimiter = ",")  



m_bia2 = np.zeros( shape=(N , 2 * M ), dtype = np.int8 )  # for nomination co-support

for r, d in join_select.iterrows():
    
    cast_value = d["cast_value"]
    rollnumber = d["rollnumber"]

    
    row_i = s_rollnumber[ s_rollnumber == rollnumber ].index[0]
    
    icpsr_index =  s_icpsr[ s_icpsr == d["icpsr"]  ].index[0]
    
    
    if cast_value == "Yes":
        col_j = icpsr_index
    if cast_value == "No":
        col_j = M + icpsr_index
    
    m_bia2[ row_i, col_j ] = 1

np.savetxt("Senator_Biadjacency_Matrix.csv", m_bia2, delimiter = ",")  


#
# Now we compute the matrix equivalent of the projection of
# The senators based on their common voting patterns for nominations.
# by multiplying the biadjacency matrix by its own transpose
# The resulting M X M matrix compares the number of common votes: yeas or nays where the senators concurred.
#
# Note that we only want to use the edges in the upper triangular half of the matrix.
# since the product is symmetric.
# -------------------------------------------------------------------------------
senator_covoting =  np.matmul( m_bia, m_bia.transpose())

#
#
# Now we draw the projected graph for the senators as an undirected
# weighted graph.  Note that we scale each senator covoting score
# by the number of nominations:  188
# to normalize the covoting from counts to fractions of votes.
# 
# Each threshold weight is entered in integer from 0-100
#
# Only include edges where covoting exceeds the threshold.
# -----------------------------------------------------------------
def make_covoting_graph( threshold_weight):

    G_covoting = nx.Graph()

    for r, d in members.iterrows():
    
        senator_id = d["icpsr"]
    
        G_covoting.add_node( senator_id, 
                 short_name = d["short_name"] ,
                 bioname = d["bioname"],
                 party = d["party"] ,
                 state_abbrev = d["state_abbrev"]
               )
        
    for i in range(M):
        for j in range(M):
            if i < j:            
                w_ij = senator_covoting[i,j]/188
            
                if w_ij > threshold_weight/100.0 :
                    G_covoting.add_edge( s_icpsr[i]  , s_icpsr[j]   ,weight= float(w_ij ) )


    graphml_file = "Senate_Covoting_K{}.graphml".format(threshold_weight) 

    nx.write_graphml( G_covoting , graphml_file)
    
    return G_covoting


G_covoting_55 = make_covoting_graph( 55  )

G_covoting_60 = make_covoting_graph( 60  )

G_covoting_65 = make_covoting_graph( 65  )


def convert_party_color(row):
    
    party = row["party"]
    
    if party == "Democrat":
        return "blue"
    if party  == "Republican":
        return "red"
    if party == "Independent":
        return "blue"
    return "Unknown"


covoting_node_colors = members.apply( lambda row :  convert_party_color(row), axis = 1 )

covoting_node_labels = { row["icpsr"] : row["short_name"] for i, row in members.iterrows()}

def plot_covoting(G_covoting):

    plt.figure(figsize = (12,10))
    plt.tight_layout()
    plt.axis("off")

    pos = nx.spring_layout(G_covoting)

    weights = [ G_covoting_55[u][v]['weight'] for u,v in G_covoting_55.edges() ]

    nx.draw_networkx_labels(G_covoting , pos = pos , labels = covoting_node_labels , font_size = 9 )

    nx.draw_networkx_edges(G_covoting , pos = pos , width =  weights, edge_color = 'gray', alpha = 0.1)

    nx.draw( G_covoting , pos = pos, with_labels = False, alpha = 0.6,  node_size = 50, node_color = covoting_node_colors )

plot_covoting(G_covoting_55)

plot_covoting(G_covoting_60)

plot_covoting(G_covoting_65)


#
# Let's example the subgraphs by party to understand
# their covoting behavior.
# -----------------------------------------------------
members_democrat = [  row["icpsr"] for d, row in members.iterrows() if row["party"] != "Republican"]

members_republican = [  row["icpsr"] for d, row in members.iterrows() if row["party"] == "Republican"]

G_democrat_covoting_55 = G_covoting_55.subgraph(members_democrat)

nx.write_graphml( G_democrat_covoting_55 , "Senate_Democrat_Covoting_K{}.graphml".format(55) )


G_republican_covoting_55 = G_covoting_55.subgraph(members_republican)

nx.write_graphml( G_republican_covoting_55 , "Senate_Republican_Covoting_K{}.graphml".format(55) )

#
#  Next we consider how nominations are considered.
#  Taking the dual approach to the senators, we ask if two nominations
#  are alike when the senators voting for their appointments are 
#  the same or nearly so.
# --------------------------------------------------------
nominee_cosupport =  np.matmul( m_bia2 , m_bia2.transpose())

#
#
# Now we draw the projected graph for the nominations as an undirected
# weighted graph.  The function arguments:  
#
#    threshold_weight:  takes a positive integer argument between 0-99
def make_cosupport_graph(threshold_weight):
    G_cosupport = nx.Graph()

    for r, d in nominations.iterrows():
    
        rollnumber = d["rollnumber"]
    
        G_cosupport.add_node( rollnumber, 
                 name = d["Name"] ,
                 dept = d["dept"],
                 position = d["Position"] ,
                 bill_number = d["bill_number"] ,
                 yea_count = d["yea_count"],
                 nay_count = d["nay_count"],
                 margin = d["yea_count"] - d["nay_count"]
               )

    for i in range(N):
        for j in range(N):
            if i < j:
            
                w_ij = nominee_cosupport[i,j]/100
            
                if w_ij > threshold_weight/100.0 :
                    G_cosupport.add_edge( s_rollnumber[i]  , s_rollnumber[j]   ,weight= float(w_ij ) )
    
    graphml_file = "Nomination_Cosupport_K{}.graphml".format(threshold_weight) 
    nx.write_graphml( G_cosupport , graphml_file)

    return G_cosupport


G_cosupport_0 = make_cosupport_graph( 0 )

G_cosupport_20 = make_cosupport_graph( 20 )

G_cosupport_40 = make_cosupport_graph( 40 )

G_cosupport_60 = make_cosupport_graph( 60 )

G_cosupport_70 = make_cosupport_graph( 70 )

G_cosupport_80 = make_cosupport_graph( 80 )

G_cosupport_90 = make_cosupport_graph( 90 )

G_cosupport_92 = make_cosupport_graph( 92 )

G_cosupport_95 = make_cosupport_graph( 95 )
