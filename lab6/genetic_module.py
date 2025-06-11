import random
# ---------- Фитнес-функция ----------
def fitness_indexed(individual, recipe_list, available_ingredients, max_total_time, nutrition_targets):
    """
    Computes the fitness score of a given individual, which is a list of indices
    into the recipe_list. The fitness score is a measure of how well the recipes
    in the individual satisfy the given constraints, such as the available
    ingredients, the maximum total time, and the nutrition targets.

    The fitness score is computed as the average of the ingredient score, rating
    score, and penalty score. The ingredient score is the proportion of
    ingredients in the recipe that are available. The rating score is the
    AggregatedRating of the recipe, normalized to the range [0, 1]. The penalty
    score is the sum of the penalties for exceeding the maximum total time and
    for deviating from the nutrition targets.

    Args:
        individual (list): A list of indices into the recipe_list.
        recipe_list (list): A list of recipes, where each recipe is a dictionary
            with the following keys:
                RecipeIngredientParts (list): A list of ingredient names.
                AggregatedRating (float): The average rating of the recipe.
                TotalTimeMinutes (int): The total time in minutes required to
                    prepare the recipe.
                [Nutrition key] (float): The amount of the given nutrition
                    substance in the recipe.
        available_ingredients (list): A list of ingredient names that are
            available.
        max_total_time (int): The maximum total time in minutes allowed.
        nutrition_targets (dict): A dictionary with the desired amounts of
            nutrition substances.

    Returns:
        float: The fitness score of the individual.
    """
    total_score = 0
    total_time = 0
    total_nutrition = {key: 0 for key in nutrition_targets}

    for idx in individual:
        recipe = recipe_list[idx]
        ingredients = recipe['RecipeIngredientParts']
        matches = sum(1 for ing in ingredients if any(a.lower() in ing.lower() for a in available_ingredients))
        ingredient_score = matches / len(ingredients) if ingredients else 0

        rating = recipe['AggregatedRating']
        rating_score = rating / 5.0 if rating > 0 else 0

        for feature in nutrition_targets:
            total_nutrition[feature] += recipe.get(feature, 0)

        total_score += ingredient_score * 0.6 + rating_score * 0.3
        total_time += recipe['TotalTimeMinutes']

    # Penalty calculations
    penalty = 0
    if total_time > max_total_time:
        penalty += 0.3 * (total_time - max_total_time) / max_total_time

    for k, v in nutrition_targets.items():
        actual = total_nutrition[k]
        penalty += 0.3 * abs(actual - v) / v

    avg_score = total_score / len(individual) if individual else 0
    return avg_score - penalty

# ---------- Кроссовер ----------
def crossover_indexed(p1, p2, min_r, max_r):
    """
    Perform crossover between two parent individuals by combining their unique indices
    and selecting a random subset of them.

    Args:
        p1 (list): The first parent individual, represented as a list of indices.
        p2 (list): The second parent individual, represented as a list of indices.
        min_r (int): The minimum number of indices to select for the child.
        max_r (int): The maximum number of indices to select for the child.

    Returns:
        list: A list of indices representing the child individual, created by
              sampling a random number of unique indices from the combined parents.
    """

    combined = list(dict.fromkeys(p1 + p2))  # Уникальные индексы
    k = min(len(combined), random.randint(min_r, max_r))
    return random.sample(combined, k)

# ---------- Мутация ----------
def mutate_indexed(individual, total_recipes, min_r, max_r, mutation_rate=0.2):
    """
    Perform mutation on an individual by randomly changing one of its indices
    with probability mutation_rate, and then either adding or removing a random
    index with probability mutation_rate/2, but only if the length of the
    individual is within the allowed range.

    Args:
        individual (list): The list of indices to mutate.
        total_recipes (int): The total number of recipes.
        min_r (int): The minimum number of indices to allow.
        max_r (int): The maximum number of indices to allow.
        mutation_rate (float, optional): The probability of mutation. Defaults to 0.2.

    Returns:
        list: The mutated individual.
    """
    new_ind = individual[:]

    if random.random() < mutation_rate:
        i = random.randint(0, len(new_ind) - 1)
        new_ind[i] = random.randint(0, total_recipes - 1)

    if len(new_ind) < max_r and random.random() < mutation_rate / 2:
        new_ind.append(random.randint(0, total_recipes - 1))

    if len(new_ind) > min_r and random.random() < mutation_rate / 2:
        del new_ind[random.randint(0, len(new_ind) - 1)]

    return list(dict.fromkeys(new_ind))  # Удаляем дубликаты

# ---------- Генетический алгоритм ----------
def genetic_algorithm_optimized_indexed(
    recipe_list, available_ingredients, max_total_time, nutrition_targets,
    population_size=50, generations=500, mutation_rate=0.7,
    min_recipes=2, max_recipes=5
):
    """
    Perform a genetic algorithm optimization on a list of recipes, using a population
    of individuals represented as lists of indices into the recipe list.

    Args:
        recipe_list (list): A list of recipes, where each recipe is a dictionary.
        available_ingredients (list): A list of ingredients available for use in the recipes.
        max_total_time (int): The maximum total time allowed for the meal plan.
        nutrition_targets (dict): A dictionary of nutrition targets for the meal plan.
        population_size (int, optional): The size of the population. Defaults to 50.
        generations (int, optional): The number of generations to run the algorithm. Defaults to 500.
        mutation_rate (float, optional): The probability of mutation. Defaults to 0.7.
        min_recipes (int, optional): The minimum number of recipes to include in the meal plan. Defaults to 2.
        max_recipes (int, optional): The maximum number of recipes to include in the meal plan. Defaults to 5.

    Returns:
        tuple: A tuple containing the best meal plan and the history of the optimization.
    """

    total_recipes = len(recipe_list)

    # Индивиды = списки индексов
    population = [
        random.sample(range(total_recipes), random.randint(min_recipes, max_recipes))
        for _ in range(population_size)
    ]
    history = []
    for _ in range(generations):
        scored_population = [
            (ind, fitness_indexed(ind, recipe_list, available_ingredients, max_total_time, nutrition_targets))
            for ind in population
        ]
        scored_population.sort(key=lambda x: x[1], reverse=True)
        top = [ind for ind, _ in scored_population[:population_size // 2]]
        history.append(scored_population[0][1])

        children = []
        for _ in range(population_size // 2):
            p1, p2 = random.sample(top, 2)
            child = crossover_indexed(p1, p2, min_recipes, max_recipes)
            child = mutate_indexed(child, total_recipes, min_recipes, max_recipes, mutation_rate)
            children.append(child)

        population = top + children

    best = max(population, key=lambda ind: fitness_indexed(ind, recipe_list, available_ingredients, max_total_time, nutrition_targets))
    return [recipe_list[i] for i in best], history