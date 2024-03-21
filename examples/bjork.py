from llmtoolz import bjork


def main():
    recipe = bjork(query="How to make a cake", validate_query=True, )
    print(recipe)
    ## Output:
    # Here is a simple and easy-to-follow recipe for a basic white cake:
    #
    # **Ingredients:**
    # 1. 1 cup (200g) white sugar
    # 2. ½ cup (113g) unsalted butter
    # 3. 2 large eggs
    # 4. 2 teaspoons (9.9 mL) vanilla extract
    # 5. 1 ½ cups (188g) all-purpose flour
    # 6. 1 ¾ teaspoons (8.4g) baking powder
    # 7. ½ cup (120 mL) milk
    #
    # **Procedure:**
    # 1. Preheat the oven to 350 degrees F (175 degrees C).
    # 2. Grease and flour a 9-inch square cake pan.
    # 3. Cream together the sugar and butter in a medium bowl.
    # 4. Beat in the eggs, one at a time, then stir in the vanilla extract.
    # 5. In a separate bowl, combine the flour and baking powder.
    # 6. Add this dry mixture to the creamed mixture and mix well.
    # 7. Stir in the milk until the batter is smooth.
    # 8. Pour the batter into the prepared pan.
    # 9. Bake in the preheated oven for 30 to 40 minutes, or until the cake springs back to the touch.
    #
    # For cupcakes, bake for 20 to 25 minutes.
    #
    # This information was gathered from the following sources:
    #
    # 1. [All Recipes](https://www.allrecipes.com/recipe/17481/simple-white-cake/)
    # 2. [WikiHow](https://www.wikihow.com/Make-a-Plain-Cake)
    # 3. [Bake From Scratch](https://bakefromscratch.com/basic-1-2-3-4-cake/)



if __name__ == "__main__":
    main()
