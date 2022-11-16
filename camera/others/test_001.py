#!/usr/bin/python
# -*- coding: utf-8 -*-

def main(root: int) -> int:
    if root <= 0:
        return 0
    print(main(root - 1))
    # print(root)
    return root


if __name__ == '__main__':
    main(10)  # 10
    pass
