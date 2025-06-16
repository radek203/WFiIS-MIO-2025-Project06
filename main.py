import yeast
import iris
import gallstone
import adult

if __name__ == "__main__":
    print("\n===================== Test 1: Yeast Dataset =====================")
    yeast.calc()

    print("\n===================== Test 2: Gallstone Dataset  =====================")
    gallstone.calc()

    print("\n===================== Test 3: Adult Dataset  =====================")
    adult.calc()

    print("\n===================== Test 4: Iris Dataset  =====================")
    iris.calc()