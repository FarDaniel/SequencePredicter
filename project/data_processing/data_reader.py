from os.path import exists


class DataReader:

    def __init__(self, separator):
        self.file = None
        self.is_file_opened = False
        self.last_number = 0.0
        self.actual_character = None
        self.separator = separator

    # Working with big data, we can't load everything in memory at once.
    def read_one(self):
        if self.file.closed:
            return
        self.last_number = 0.0
        self.actual_character = self.file.read(1)
        while self.actual_character != self.separator:
            if self.actual_character.isnumeric():
                # Shifting digits by one
                self.last_number *= 10
                # Adding in the next digit (we know that it's exactly one digit)
                self.last_number += float(self.actual_character)
            elif self.actual_character != self.separator:
                # We hit EOF (end of file)
                if self.actual_character == "":
                    self.close_file()

                    # The last number hasn't got a separator after itself,
                    # which means that we can return with last_number, when we hit EOF.
                    return self.last_number
                raise RuntimeError("Unexpected character occurred in dataset!")

            self.actual_character = self.file.read(1)
        return self.last_number

    def get_numbers(self, cnt):
        numbers = []
        for i in range(cnt):
            number = self.read_one()
            if number is not None:
                numbers.append(number)
        return numbers

    # In case we don't want to read until EOF
    def close_file(self):
        self.file.close()
        self.is_file_opened = False

    def open_file(self, file):
        if self.is_file_opened:
            self.close_file()
        if exists(file):
            self.file = open(file, "r")
            self.is_file_opened = True
            print('A feldolgozás elkezdődött!')
        else:
            print('A file nem létezik!')
