import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # Загружаем твой CSV
    df = pd.read_csv("ft_hparams_results2.csv")
    
    # Настройка стиля
    sns.set_theme(style="whitegrid")
    
    # 1. График влияния Rank (Ранга) на Exact Match
    # Берем только те строки, где менялся Rank (Config 1, 11, 0, 2, 3 - это 4, 8, 16, 32, 64)
    # Предполагаем, что Note помогает фильтровать, или просто берем по ID
    rank_df = df[df['Note'].str.contains('Rank') | (df['Note'] == 'Baseline')].sort_values('LoRA R')
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=rank_df, x='LoRA R', y='Exact Match', marker='o', label='Exact Match', color='blue')
    sns.lineplot(data=rank_df, x='LoRA R', y='Slot-F1', marker='s', label='Slot-F1', color='green')
    plt.title('Влияние ранга LoRA (Rank) на качество')
    plt.xlabel('LoRA Rank')
    plt.ylabel('Score')
    plt.legend()
    plt.savefig('plot_rank_impact.png')
    print("График Rank сохранен.")

    # 2. Итоговое сравнение всех конфигов (Bar Plot)
    plt.figure(figsize=(12, 6))
    barplot = sns.barplot(data=df, x='Config ID', y='Exact Match', palette='viridis')
    plt.title('Сравнение всех экспериментов (Exact Match)')
    plt.ylim(0.75, 0.82) # Сужаем ось Y, чтобы видеть разницу
    plt.ylabel('Exact Match Score')
    
    # Добавляем цифры над столбиками
    for p in barplot.patches:
        barplot.annotate(format(p.get_height(), '.4f'), 
                         (p.get_x() + p.get_width() / 2., p.get_height()), 
                         ha = 'center', va = 'center', 
                         xytext = (0, 9), 
                         textcoords = 'offset points')
                         
    plt.savefig('plot_all_configs.png')
    print("График сравнения сохранен.")

if __name__ == "__main__":
    main()