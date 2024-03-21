import torch
key_words_name = ["pleomorphism",["nuclei","nuclear","degree of crowding"],"polarity",["mitosis","mitotic"],"nucleoli"]

key_words = [
     
    ['severe','moderate','normal','mild'], #Sentence 0 Pleomorphism
    
    [['no signs','no nuclear crowding'],['normal','normally'],'mild','severe',['moderate','moderately']], #Sentence 1 Nuclei
    
    ['normal',['not lost','negligibly lost','no loss'], # Sentence 2: POLARITY
     ['is completely lost','complete lack of','lack of cellular polarity'],
     ['some degree','not completely lost','partially lost']],
    
    ['rare',['are frequently','is frequent','are frequent'],'infrequent'], # Sentence 3: Mitosis
    
    ['inconspicuous',['are prominent','prominent nucleoi','is prominent'],'rare'], # Sentence 4 Nucleoli
]


# key_word_weights = [
#     torch.tensor([3.45, 0.61, 3.5, 0.42, 2.46]),
#     torch.tensor([2.4, 1.56, 0.45, 3.23, 0.55, 1.67]),
#     torch.tensor([4.01, 1.09, 0.7, 1.18, 0.65]),
#     torch.tensor([0.38, 2.39, 1.82, 2.38]),
#     torch.tensor([0.53, 1.11, 1.56, 1.77])
# ]

key_word_weights = [
    torch.tensor([10.0, 1.0, 10.0, 1.0, 5.0]),
    torch.tensor([4.0, 2.0, 1.0, 10.0, 0.55, 2.0]),
    torch.tensor([8.0, 2.0, 1.0, 2.5, 0.9]),
    torch.tensor([1.0, 8.0, 3.0, 8.0]),
    torch.tensor([1.0, 3.0, 5.0, 6.0])
]