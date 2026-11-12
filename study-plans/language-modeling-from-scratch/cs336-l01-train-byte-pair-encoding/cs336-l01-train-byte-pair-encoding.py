from collections import Counter

def train_bpe(corpus: list[str], vocab_size: int) -> dict:
    """
    Returns a dictionary of learned vocab entries and ordered merges.
    Parameters
    ----------
    corpus : list[str]
        Список строк для обучения.
        Каждая строка обрабатывается независимо.

    vocab_size : int
        Целевой размер словаря, включая начальные 256 токенов.

    Returns
    -------
    dict
        {
            "vocab": [
                [new_token_id, byte_values],
                ...
            ],
            "merges": [
                [left_id, right_id, new_id],
                ...
            ]
        }
    """
    # ID 0..255 соответствуют одиночным байтам
    token_bytes = {
        i: bytes([i])
        for i in range(256)
    }

    

    # Начальная токенизация корпуса
    sequences = [
        list(s.encode("utf-8"))
        for s in corpus
    ]

    # Результирующие структуры
    vocab = []
    merges = []

    

    # --------------------------------------------------
    # 2. Количество необходимых объединений
    # --------------------------------------------------

    num_merges = max(0, vocab_size - 256)

    

    # --------------------------------------------------
    # 2. Количество необходимых объединений
    # --------------------------------------------------

    num_merges = max(0, vocab_size - 256)

    # --------------------------------------------------
    # 3. Основной цикл обучения
    # --------------------------------------------------

    for _ in range(num_merges):

        # ----------------------------------------------
        # 3.1. Подсчёт частот соседних пар
        # ----------------------------------------------

        pair_counts = Counter()

        for seq in sequences:
            for i in range(len(seq) - 1):
                pair = (seq[i], seq[i + 1])
                pair_counts[pair] += 1

        # Если пар больше нет, останавливаем обучение
        if not pair_counts:
            break

        # ----------------------------------------------
        # 3.2. Выбор наиболее частой пары
        # ----------------------------------------------

        best_pair = max(
            pair_counts,
            key=lambda pair: (
                pair_counts[pair],
                token_bytes[pair[0]],
                token_bytes[pair[1]]
            )
        )

        left_id, right_id = best_pair

        # ----------------------------------------------
        # 3.3. Создание нового токена
        # ----------------------------------------------

        new_id = 256 + len(vocab)

        new_bytes = (
            token_bytes[left_id] +
            token_bytes[right_id]
        )

        token_bytes[new_id] = new_bytes

        # print(new_id, token_bytes[left_id], token_bytes[right_id], new_bytes, list(new_bytes))

        vocab.append([
            new_id,
            list(new_bytes)
        ])

        merges.append([
            left_id,
            right_id,
            new_id
        ])

        # ----------------------------------------------
        # 3.4. Замена выбранной пары в корпусе
        # ----------------------------------------------

        new_sequences = []

        for seq in sequences:

            new_seq = []
            i = 0

            while i < len(seq):

                # Проверяем совпадение пары
                if (
                    i < len(seq) - 1
                    and seq[i] == left_id
                    and seq[i + 1] == right_id
                ):
                    new_seq.append(new_id)

                    # Пропускаем оба объединённых токена
                    i += 2

                else:
                    new_seq.append(seq[i])
                    i += 1

            new_sequences.append(new_seq)

        sequences = new_sequences

    # --------------------------------------------------
    # 4. Возвращаем результат
    # --------------------------------------------------

    return {
        "vocab": vocab,
        "merges": merges
    }
