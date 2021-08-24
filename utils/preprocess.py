import random

if __name__ == '__main__':
    with open("../binary/sudoku.csv") as csv_file:
        data = csv_file.readlines()
        header = data[:1]
        data = data[1:]

        random.shuffle(data)

        with open("../binary/sudoku_test.csv", mode="w") as file:
            file.writelines(header + data[:10000])

        with open("../binary/sudoku_validate.csv", mode="w") as file:
            file.writelines(header + data[10000:20000])

        with open("../binary/sudoku_train.csv", mode="w") as file:
            file.writelines(header + data[20000:])
