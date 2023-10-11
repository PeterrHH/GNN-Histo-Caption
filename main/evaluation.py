# -*- coding=utf-8 -*-
# author: w61
# Test for several ways to compute the score of the generated words.

'''
Need java 1.8
Package:
pycocoevalcap
'''
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice
import os
import sys
import ssl
class Scorer():
    def __init__(self,ref,gt):
        self.ref = ref
        self.gt = gt
        print('setting up scorers...')
        self.word_based_scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(),"METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),]
            # (Spice(), "SPICE"),]


        
    def compute_scores(self):
        total_scores = {
            "Bleu1":[],
            "Bleu2":[],
            "Bleu3":[],
            "Bleu4":[],
            "METEOR":[],
            "ROUGE_L":[],
            "CIDEr":[]
            # "SPICE":[]
        
        }

        for scorer, method in self.word_based_scorers:
            #print('computing %s score...'%(scorer.method()))
            # print(scorer)
            # print(method)
            score, scores = scorer.compute_score(self.ref, self.gt)
    
            if type(method) == list:
            #    for sc, scs, m in zip(score, scores, method):
                    # print("----------Bleu----------")
                    # print("%s: %0.3f"%(m, sc))
                total_scores["Bleu1"].append(score[0])
                total_scores["Bleu2"].append(score[1])
                total_scores["Bleu3"].append(score[2])
                total_scores["Bleu4"].append(score[3])
            else:
                # print("----------Other----------")
                # print("%s: %0.3f"%(method, score))
                # print(f"method is {method} total is {total_scores}")
                total_scores[method].append(score)
        # print(total_scores)
        
        # print('*****DONE*****')

        return total_scores
    def compute_scores_iterative(self):


        total_scores = {
            "Bleu1":[],
            "Bleu2":[],
            "Bleu3":[],
            "Bleu4":[],
            "METEOR":[],
            "ROUGE_L":[],
            "CIDEr":[],
            "SPICE":[]
        
        }

        for key in self.ref:
            
            curr_ref = {key:self.ref[key]}
            curr_gt = {key:self.gt[key]}

            for scorer, method in self.word_based_scorers:
                #print('computing %s score...'%(scorer.method()))
                # print(scorer)
                # print(method)
                #print(curr_gt)
                score, _ = scorer.compute_score(curr_ref, curr_gt)
                # print(score)
                if type(method) == list:
                    # for sc, scs, m in zip(score, scores, method):
                        # print("----------Bleu----------")
                        # print("%s: %0.3f"%(m, sc))
                    total_scores["Bleu1"].append(score[0])
                    total_scores["Bleu2"].append(score[1])
                    total_scores["Bleu3"].append(score[2])
                    total_scores["Bleu4"].append(score[3])
                else:
                    # print("----------Other----------")
                    # print("%s: %0.3f"%(method, score))
                    # print(f"method is {method} score is {score}")
                    total_scores[method].append(score)
                # print(score)
        
        # print('*****DONE*****')
        # print(total_scores)
        return total_scores
        # for key,value in total_scores.items():
        #     print('{}:{}'.format(key,value))

if __name__ == '__main__':
    import numpy as np
    ssl._create_default_https_context = ssl._create_unverified_context # Use it to solve SSL 
    def unpack_score(scores):
        bleu = scores["Bleu"]
        meteor = scores["METEOR"]
        rouge = scores["ROUGE_L"]
        cider = scores["CIDEr"]
        #spice = scores["SPICE"]
        return bleu[0], bleu[1], bleu[2], bleu[3], meteor, rouge, cider
    ref_dict = {
        '1': ['Walk down the steps and stop at the bottom. ', 'Go down the stairs and wait at the bottom.','Once at the top of the stairway, walk down the spiral staircase all the way to the bottom floor. Once you have left the stairs you are in a foyer and that indicates you are at your destination.'],
        '2': ['It is a cat.','It is a cat.','cat over there.'],
        '3': ['i know it, you know it','I do now know what this mean','THis is good']
    }

    gen_dict = {
        '1': ['go down the stairs all the way and stop at the bottom .'],
        '2': ['It is a cat.'],
        '3': ['I know how good this is']
    }

    ref = {
           '1':['Slight variability in nuclear size shape and outline consistent with mild pleomorphism . There is a severe degree of crowding . Polarity is completely lost . Mitosis is frequent throughout the tissue . Prominent nucleoli are easily identified in low magnification scanning . High grade .',
                'Mild pleomorphism is present . There is a severe degree of crowding . Architecturally the cells show complete lack of polarity toward the surface urothelium . Mitosis is frequent throughout the tissue . The nucleoli of nuclei are prominent . High grade .', 
                'Mild pleomorphism is present . Nuclei are severely crowded together . There is marked disorganization and lack of cellular polarity toward the surface urothelium . Mitosis is frequent . Nucleoli is prominent . High grade .', 
                'Mild pleomorphism and cytologic atypia is present . The nuclei are crowded to a severe degree . Polarity is completely lost . There are frequent mitotic figures throughout the tissue . The nucleoli of nuclei are prominent . High grade .', 
                'Nuclear features show mild pleomorphism . There is a severe degree of crowding . Architecturally the cells show complete lack of polarity toward the surface urothelium . Mitotic figures including the atypical forms are frequently seen in all levels of the urothelium . Nucleoli is prominent . High grade .']}
    gen = {
           '1':['mild pleomorphism . there are and prominent nucleoli are not observed or exceedingly rare and limited to .']}
    



    scorer = Scorer(ref,gen)
    tol_scores = scorer.compute_scores()
    print(tol_scores)
    print("-------TOTAL SCORE ABOVE----------")
    # scores = scorer.compute_scores_iterative()
    # for idx,value in scores.items():
    #     mean = np.mean(value)
    #     std = np.std(value)
    #     print(f"for idx = {idx} mean is {mean} std is {std}")

    # Calculate mean and variance for each score
#  
    # pack = unpack_score(scores)
    # print(pack)
    # for i in range(len(pack)):
    #     print(pack[i])
