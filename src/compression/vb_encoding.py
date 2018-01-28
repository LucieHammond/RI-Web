
def byte_encode(id, c=1):
    if id < 128:
        return [c * 128 + id]
    else:
        rest = id % 128
        quotient = id // 128
        return byte_encode(quotient, c=0) + [c * 128 + rest]


def byte_decode(byte, c=1):
    if len(byte) == 1:
        return byte[0] - c * 128
    else:
        return 128 * byte_decode(byte[:-1], c=0) + byte[-1] - c * 128


if __name__ == "__main__":
    # 2.3 Création d'un index inversé compressé
    print("2.3 Creation of Compressed Inversed Index\n")

    # Variable Byte Encoding
    print("--- Variable Byte Encoding ---")

    tests = [9, 137, 1994, 10334, 998008]
    for test in tests:
        b = byte_encode(test)
        num = byte_decode(b)
        print('test: %i ---encode---> %s ---decode---> %i' % (test, str(b), num))

    print("\nPour voir cette méthode de compression à l'oeuvre lors de la création des indexes \net lors des "
          "recherches d'information, des sections ont été ajoutés dans :")
    print("- index_builder")
    print("- bool_search")
    print("- vect_search")
