
#import  the necessary libraries
import pandas as pd

from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

#model
model_name = "t5-small"
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)

df = pd.read_csv('calculations.csv')
#print first 40 samples
df.head(40)

questions = list(df['input'])
answers = list(df['output'])

inputs = tokenizer(questions, return_tensors = 'pt', padding= True, truncation=True)
labels = tokenizer(answers, return_tensors = 'pt', padding=True, truncation=True).input_ids

#using Adam as a optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
model.train()

#Try to increase the epochs for better results
for epoch in range(5):
  optimizer.zero_grad()
  outputs = model(**inputs, labels = labels)
  loss = outputs.loss
  loss.backward()
#updating the weights and bias
  optimizer.step()
  print(f"Epoch {epoch} Loss: {loss.item()}")

#evaluating the model
model.eval()

#predection
new_question = "Multiply 9 with 9"
input_ids = tokenizer(new_question, return_tensors='pt').input_ids
with torch.no_grad():
  generated_ids = model.generate(input_ids)
  structured_output = tokenizer.decode(generated_ids[0],skip_special_tokens=True)
  print(structured_output)

def calculation(structured_output):
    operation = structured_output.split(" ")
    
    if(operation[0]== 'sum' or operation[0]== 'Add' or operation[0]== '+' ):
      print(int(operation[1]) + int(operation[2]))
    elif(operation[0]== 'subtract' or operation[0]== 'Minus' or operation[0]== '-'):
      print(int(operation[2]) - int(operation[1]))
    elif(operation[0]=='multiply' or operation[0]== '*'):
      print(int(operation[1]) * int(operation[2]))
