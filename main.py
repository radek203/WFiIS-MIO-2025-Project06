import yeast
import iris
import gallstone

if __name__ == "__main__":
    print("\n===================== Test 1: Yeast Dataset =====================")
    yeast.calc()

    print("\n===================== Test 2: Iris Dataset  =====================")
    iris.calc()

    print("\n===================== Test 3: Gallstone Dataset  =====================")
    gallstone.calc()