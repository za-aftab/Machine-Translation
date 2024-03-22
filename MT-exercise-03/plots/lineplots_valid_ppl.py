import pandas as pd
import matplotlib.pyplot as plt

valid_ppl_df = pd.read_excel('MT_Validation_ppl.xlsx', sheet_name='Validation Perplexity')

x = valid_ppl_df['Validation Perplexity']
y1 = valid_ppl_df['Dropout 0']
y2 = valid_ppl_df['Dropout 0.1']
y3 = valid_ppl_df['Dropout 0.5']
y4 = valid_ppl_df['Dropout 0.7']
y5 = valid_ppl_df['Dropout 0.9']

plt.plot(x, y1, label='Dropout 0')
plt.plot(x, y2, label='Dropout 0.1')
plt.plot(x, y3, label='Dropout 0.5')
plt.plot(x, y4, label='Dropout 0.7')
plt.plot(x, y5, label='Dropout 0.9')

plt.title('Validation Perplexity')
plt.xlabel('Epochs')
plt.ylabel('Validation Perplexity')
plt.legend()
plt.show()