import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# Створюємо вхідні змінні
food_quality = ctrl.Antecedent(np.arange(0, 11, 1), 'food_quality')
service = ctrl.Antecedent(np.arange(0, 11, 1), 'service')
ambience = ctrl.Antecedent(np.arange(0, 11, 1), 'ambience')

# Створюємо вихідну змінну
satisfaction = ctrl.Consequent(np.arange(0, 11, 1), 'satisfaction')

# Визначаємо функції належності для вхідних змінних
food_quality.automf(names=['poor', 'average', 'excellent'])
service.automf(names=['poor', 'average', 'excellent'])
ambience.automf(names=['poor', 'average', 'excellent'])

# Визначаємо функції належності для вихідної змінної
satisfaction['very_low'] = fuzz.trimf(satisfaction.universe, [0, 0, 2.5])
satisfaction['low'] = fuzz.trimf(satisfaction.universe, [0, 2.5, 5])
satisfaction['medium'] = fuzz.trimf(satisfaction.universe, [2.5, 5, 7.5])
satisfaction['high'] = fuzz.trimf(satisfaction.universe, [5, 7.5, 10])
satisfaction['very_high'] = fuzz.trimf(satisfaction.universe, [7.5, 10, 10])

# Розширений набір правил
rules = [
    # Правила для дуже низької задоволеності
    ctrl.Rule(food_quality['poor'] & service['poor'] & ambience['poor'],
              satisfaction['very_low']),

    # Правила для низької задоволеності
    ctrl.Rule(food_quality['poor'] & service['poor'] & ambience['average'],
              satisfaction['low']),
    ctrl.Rule(food_quality['poor'] & service['average'] & ambience['poor'],
              satisfaction['low']),
    ctrl.Rule(food_quality['average'] & service['poor'] & ambience['poor'],
              satisfaction['low']),

    # Правила для середньої задоволеності
    ctrl.Rule(food_quality['average'] & service['average'] & ambience['average'],
              satisfaction['medium']),
    ctrl.Rule(food_quality['excellent'] & service['poor'] & ambience['poor'],
              satisfaction['medium']),
    ctrl.Rule(food_quality['poor'] & service['excellent'] & ambience['excellent'],
              satisfaction['medium']),

    # Правила для високої задоволеності
    ctrl.Rule(food_quality['excellent'] & service['average'] & ambience['average'],
              satisfaction['high']),
    ctrl.Rule(food_quality['average'] & service['excellent'] & ambience['excellent'],
              satisfaction['high']),
    ctrl.Rule(food_quality['excellent'] & service['excellent'] & ambience['average'],
              satisfaction['high']),

    # Правила для дуже високої задоволеності
    ctrl.Rule(food_quality['excellent'] & service['excellent'] & ambience['excellent'],
              satisfaction['very_high']),
]

# Створюємо систему керування
satisfaction_ctrl = ctrl.ControlSystem(rules)
satisfaction_sim = ctrl.ControlSystemSimulation(satisfaction_ctrl)


def calculate_satisfaction(food_q, serv, amb):
    """Обчислює рівень задоволеності на основі вхідних параметрів."""
    satisfaction_sim.input['food_quality'] = food_q
    satisfaction_sim.input['service'] = serv
    satisfaction_sim.input['ambience'] = amb
    satisfaction_sim.compute()
    return satisfaction_sim.output['satisfaction']


def plot_membership_functions():
    """Візуалізує функції належності для всіх змінних."""
    fig, (ax0, ax1, ax2, ax3) = plt.subplots(nrows=4, figsize=(10, 12))

    # Встановлюємо кольори для різних функцій належності
    colors = ['red', 'blue', 'green']

    # Візуалізація функцій належності для food_quality
    for term, color in zip(['poor', 'average', 'excellent'], colors):
        ax0.plot(food_quality.universe, food_quality[term].mf, color, label=term)
    ax0.set_title('Food Quality Membership Functions')
    ax0.legend()

    # Аналогічно для service
    for term, color in zip(['poor', 'average', 'excellent'], colors):
        ax1.plot(service.universe, service[term].mf, color, label=term)
    ax1.set_title('Service Membership Functions')
    ax1.legend()

    # Аналогічно для ambience
    for term, color in zip(['poor', 'average', 'excellent'], colors):
        ax2.plot(ambience.universe, ambience[term].mf, color, label=term)
    ax2.set_title('Ambience Membership Functions')
    ax2.legend()

    # Для satisfaction використовуємо 5 різних кольорів
    satisfaction_colors = ['red', 'orange', 'yellow', 'green', 'blue']
    for term, color in zip(['very_low', 'low', 'medium', 'high', 'very_high'], satisfaction_colors):
        ax3.plot(satisfaction.universe, satisfaction[term].mf, color, label=term)
    ax3.set_title('Satisfaction Membership Functions')
    ax3.legend()

    # Налаштування для всіх підграфіків
    for ax in [ax0, ax1, ax2, ax3]:
        ax.grid(True)
        ax.set_xlabel('Scale')
        ax.set_ylabel('Membership')

    plt.tight_layout()
    plt.show()


def test_system():
    """Тестує систему на різних комбінаціях вхідних параметрів."""
    test_cases = [
        (1, 1, 1),  # Все погано
        (3, 3, 3),  # Нижче середнього
        (5, 5, 5),  # Середні значення
        (7, 7, 7),  # Вище середнього
        (9, 9, 9),  # Все відмінно
        (9, 3, 3),  # Хороша їжа, погане обслуговування і атмосфера
        (3, 9, 9),  # Погана їжа, хороше обслуговування і атмосфера
        (9, 9, 3),  # Хороша їжа і обслуговування, погана атмосфера
    ]

    print("\nРезультати тестування системи:")
    print("-" * 60)
    print("Якість їжі | Обслуговування | Атмосфера | Задоволеність")
    print("-" * 60)

    for food_q, serv, amb in test_cases:
        result = calculate_satisfaction(food_q, serv, amb)
        print(f"{food_q:^10} | {serv:^14} | {amb:^9} | {result:^12.2f}")
    print("-" * 60)


if __name__ == "__main__":
    # Візуалізуємо функції належності
    plot_membership_functions()

    # Проводимо тестування системи
    test_system()