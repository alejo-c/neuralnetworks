import os, sys


def remove_blanks(s):
    return s.replace(" ", "").replace("\t", "")


def is_valid_line(line):
    real_line = remove_blanks(line)
    return real_line[0] not in ("#", "\n") and len(real_line) > 0


def main():
    filename = sys.argv[1]

    if os.path.exists(filename):
        with open(filename, "r") as file:
            lines = file.readlines()
            valid_lines = [line for line in lines if is_valid_line(line)]
            print(f"The file has {len(valid_lines)} lines")
    else:
        print(f"File [{filename}] does not exist in the path")


if __name__ == "__main__":
    main()
