# -*- coding: utf-8 -*-
"""
Created on Tue May 26 18:35:38 2020

@author: Murat Cihan Sorkun

Genetic Algorithm for Feature Selection
"""


from sklearn.model_selection import  cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn import linear_model
import random
import numpy as np 
import matplotlib.pyplot as plt


def crossover(crom1,crom2):

    ramdom_selections=np.random.randint(1, 3, size=len(crom1))
    child1=np.zeros(len(crom1)).astype(int)
    child2=np.zeros(len(crom1)).astype(int)
    for idx, selection in enumerate(ramdom_selections):
        if(selection==1):
            child1[idx]=crom1[idx]
            child2[idx]=crom2[idx]
        else:
            child1[idx]=crom2[idx]
            child2[idx]=crom1[idx]
            
    return child1,child2

#Mutation on a single gene
def mutation_single(crom, mut_ratio):

    if(mut_ratio>=random.random()):
       is_mut=True
       mut_idx = np.random.randint(0, len(crom))
       if(crom[mut_idx]==1):
           crom[mut_idx]=0
       else:
           crom[mut_idx]=1         
    else:
       is_mut = False
       
    return crom, is_mut

#Mutation on a multiple genes
def mutation_multi(crom, mut_ratio):

    is_mut = False
    for idx, gene in enumerate(crom):
       if(mut_ratio>=random.random()):
           is_mut=True
           if(crom[idx]==1):
               crom[idx]=0
           else:
               crom[idx]=1 
       
    return crom, is_mut


def new_generation(population,names_list,sorted_idxs,generation,mut_ratio=0.5):
    """
    select best %10 to next generation  
    select random 60% + top 20% as parents
        get 80% child   
    generate new bloods for the remaning (expected 10%) 
    """    

    new_population=[]
    new_names_list=[]
    #select best %10 to next
    best_size=round(len(population)/10)
    for idx in range(best_size):
            new_population.append(population[sorted_idxs[idx]])
            new_names_list.append(names_list[sorted_idxs[idx]])
                  
    #select random 60% + top 20%
    removed_size=round(len(population)/5)
    if(removed_size%2==1):
        removed_size=removed_size-1
    random_removed_idxs=random.sample(range(removed_size, len(population)), removed_size)
    random_selected_parents=np.delete(population, random_removed_idxs, 0)  
    random_match_parent_idxs=random.sample(range(0, len(random_selected_parents)), len(random_selected_parents))
    
    #generate 80% children and apply mutation
    for i in range(0,len(random_match_parent_idxs),2):        
        random_child1,random_child2 = crossover(random_selected_parents[random_match_parent_idxs[i]],random_selected_parents[random_match_parent_idxs[i+1]])    
        random_child1,is_mut1 = mutation_single(random_child1,mut_ratio)
        random_child2,is_mut2 = mutation_single(random_child2,mut_ratio)
        new_population.append(random_child1)
        new_population.append(random_child2)
        
        if(is_mut1):
            new_names_list.append("G("+str(generation)+")-"+str(i)+"[M]")
        else:
            new_names_list.append("G("+str(generation)+")-"+str(i))
 
        if(is_mut1):
            new_names_list.append("G("+str(generation)+")-"+str(i+1)+"[M]")
        else:
            new_names_list.append("G("+str(generation)+")-"+str(i+1))

       
    #generate new bloods for the remaning (expected 10%)    
    new_blood_size = len(population)-len(new_population) 
    for i in range(new_blood_size):    
       new_population.append(np.random.randint(0, 2, size=len(population[0])))
       new_names_list.append("G("+str(generation)+")-"+str(i+len(new_population))+"[NB]")
        
    return new_population,new_names_list



def select_features(data,target,population_size,num_of_generations,mut_ratio,task_type="clf",verbose="1"):
    """
    task_type: "clf" for classfication and "reg" for regression
    verbose: "0" for silence and "1" for show each generation
    """
    if(population_size%2!=0):
        population_size=population_size+1
        print("Population size increased to:",population_size)
    num_of_genes = data.shape[1]
    population = np.random.randint(0, 2, size=(population_size, num_of_genes))
    names_list=['G(0)-{}'.format(i) for i in range(0, population_size)]

    best_scores=[]
    avg_scores=[]
    #Classification or Regression
    if(task_type=="clf"):
        model= linear_model.LogisticRegression(solver="liblinear", multi_class="ovr")    
        reverse_sort=True 
    else:
        model= linear_model.LinearRegression()  
        # model= DecisionTreeRegressor() 
        # model= RandomForestRegressor()     
        reverse_sort=True 
    
    for generation in range(num_of_generations):
        
        if(verbose>0):
            print("\nGeneration ",generation)
        
        fitness_list=[]
        for idx,instance in enumerate(population):
                    
            zero_ids = np.where(instance == 0)[0]
            instance_data=np.delete(data, zero_ids, axis=1)
            score = cross_val_score(model, instance_data, target, cv=5).mean()
            fitness_list.append(score)
            if(verbose>0):
                print(instance,"\tScore:",round(score, 3),"\t",names_list[idx])
            
      
        best_scores.append(max(fitness_list))
        avg_scores.append(np.mean(fitness_list))
            
        #sort results by indexes        
        idxs = list(zip(*sorted([(val, i) for i, val in enumerate(fitness_list)],reverse=reverse_sort)))[1]
        
        #get new generation
        population,names_list=new_generation(population,names_list,idxs,generation,mut_ratio=mut_ratio)
    
    if(verbose>0):
        fig, ax = plt.subplots(figsize=(10,6))
        ax.plot(best_scores,label="best score")
        if(task_type=="clf"):
            ax.plot(avg_scores,label="average score")
        ax.legend()
        ax.set_xlabel('Generation')
        ax.set_ylabel('Score')
        
    return population[0]


def transform_data(data,selected_features):
    """
    Removes the features which are not selected (indicated by 0)
    """
    zero_ids = np.where(selected_features == 0)[0]
    clean_data=np.delete(data, zero_ids, axis=1)
    return clean_data



