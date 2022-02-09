import numpy as np
import torch
import copy


# cuda or cpu
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('this computation is running on {}'.format(device))
device_name = str(device)[:3]

# open ascii txt:
filename = "data/romantische_lieder_hermannhesse.txt"
text_corpus = open(filename, 'r', encoding='utf-8').read()
raw_text = text_corpus


# data cleansing: replace unnecessary characters and line breaks
# (in prediction too much linebreaks after each other leads to linebreak as prediction)
stanza = copy.copy(text_corpus)

sub_numberation = ['I.', 'II.', 'III.', 'IV.', 'V.', 
            'VI.','I\n\n', 'II\n\n', 'III\n\n', 'IV\n\n', 'V\n\n', 'VI\n\n', 'I\n\n']
for num in sub_numberation:
    stanza = stanza.replace(num, "\n")
    
end_sentence_char = ['_',':', '$', '*', '/','(',')', "'", '-', '^', '»', '«']
for char in end_sentence_char:
    stanza = stanza.replace(char, "")
    
# delete citization from text:
zitat_in = 'Ich habe den Fuß an jene Stelle des Lebens gesetzt,'
zitat_out = 'Dante'
inso = stanza.find(zitat_in)
outso = stanza.find(zitat_out)
stanza = stanza.replace(stanza[inso:outso+6], '')

stanza = stanza.replace('\n\n\n\n','\n')
stanza = stanza.replace('\n\n\n','\n')

# correction of some subtitle in order to match poemtitle - poem schematic
stanza = stanza.replace('Villalilla.\n', 'Villalilla.')
stanza = stanza.replace('Berceuse.\n\n', '\nBerceuse.\n')
stanza = stanza.replace('Grande valse.\n\n', 'Grande valse.\n')
stanza = stanza.replace('Du aber.\n\n', '\nDu aber.\n')
stanza = stanza.replace('Ich fragte Dich.\n\n', '\nIch fragte Dich.\n')
stanza = stanza.replace('Wenn doch mein Leben \n\n', '\nWenn doch mein Leben.\n')
stanza = stanza.replace('So ziehen Sterne \n\n', '\nSo ziehen Sterne.\n')
stanza = stanza.replace('So schön bist Du!\n\n', 'So schön bist Du!\n')
num_stanza = stanza.count('\n\n')


f_save = "data/rl_stanza.txt"
text_file = open(f_save, "w")
n = text_file.write(stanza)
text_file.close()

print('poem prepared and saved in '+f_save)