# Author: Le Anh Hao
# MSSV: 2251050023

import matplotlib.pyplot as plt
import numpy as np

# ==================== FOR LOOPS =======================
# # Bai 1
# yourName = input("Enter your name: ")
# for i in yourName:
#     print(i, end=' ')

# Bai 2
# start, end = input("Nhap so bat dau va ket thuc: ").split()
# for i in range(int(start), int(end)):
#     if i % 2 != 0:
#         print(i, end=' ')

# Bai 3
# tong = 0
# start, end = input("Nhap so bat dau va ket thuc: ").split()
# for i in range(int(start), int(end)):
#     if i % 2 != 0:
#         tong += i
# print(f"Tong cac so le tu {start} den {end} la: {tong}")

# Bai 4
# mydict={"a": 1, "b": 2, "c": 3, "d": 4}

# for i in mydict:
#     # print(i)
#     # print(mydict[i])
#     print(i, ": ", mydict[i])

# Bai 5
# zip() gom cac list, tuple thành các tuple
# courses=[131,141,142,212]
# names=["Maths", "Physics", "Chem", "Bio"]

# result = zip(courses, names)
# # for c, n in result:
# #     print(c, ": ", n)
# print(list(result))

# Bai 6
# nguyenAm = "ueoai"
# str = "jabbawocky"

# print("Directly")
# for c in str:
#     if c not in nguyenAm:
#         print(c, end=" ")

# print("\n\nUsing continue")
# for c in str:
#     if c in nguyenAm:
#         continue
#     print(c, end=" ")

# Bai 7
# for a in range(-2, 3):
#     try:
#         print(10/a)
#     except ZeroDivisionError:
#         print("Can’t divided by zero")

# Bai 8
# ages=[23,10,80]
# names=["Hoa","Lam","Nam"]

# result = zip(ages, names)
# Sap xep theo tuoi
# sorted_result = sorted(result, key=lambda x: x[0])
# # print(sorted_result)
# for a, n in sorted_result:
#     print(a, ": ", n)


# =============== READ FILES ======================
# Bai 9
# with open giup tu dong dong file sau khi thuc hien xong
# Khi mo 1 file khong ton tai -> nen ra exception
# with open("firstname.txt", "r") as f:
#     print(f.read())
# Doc tung dong vs readlines() hoac for loop

# =============== DEFINE  A FUNCTION ======================
# Bai 1
# *args: tuple
# **args: dict
# def twoSum(*args):
#     return sum(args)
# print(twoSum(3, 4))

# Bai 2
# M = np.array([[1,2,3], [4,5,6], [7,8,9]])
# V =  np.array([1,2,3])

# def rank(matrix):
#     return np.linalg.matrix_rank(matrix)

# print(rank(V))

# Bai 3
# M = np.array([[1,2,3], [4,5,6], [7,8,9]])
# a = 3
# def plus(M, a):
#     new_M = M + a
#     return new_M
# print(plus(M, a))


# Bai 4
# M = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# V = np.array([1, 2, 3])

# def chuyenVi(matrix):
#     return matrix.T

# print(chuyenVi(M))


# =============== MATH ======================
# Bai 5
# x = np.array([2, 7])

# # Tinh chuan cua x
# # sqrt(2^2 + 7^2)
# norm_x = np.linalg.norm(x)

# # Chuan hoa vector x
# normalized_x = x / norm_x

# # In kết quả
# print("Norm của x:", norm_x)
# print("Chuẩn hóa của x:", normalized_x)

# Bai 6
# A = np.array([10,15])
# B = np.array([8,2])
# C = np.array([1,2,3])

# # A + B, A - B duoc vi 2 ma tran co cung kich thuoc
# print(A+B)
# print(A-B)
# # A - C khong duoc vi 2 ma tran khac kich thuoc
# print(A-C)

# Bai 7
# A = np.array([10,15])
# B = np.array([8,2])

# print("Tich vo huong cua 2 ma tran A va B la: ", np.dot(A, B))

# Bai 8
# A = np.array([[2,4,9],[3,6,7]])
# print("Matrix A:", np.linalg.matrix_rank(A))
# # Hang 1 cot 2
# print("Get the value 7 in A: ", A[1][2])
# print("Cot thu 2 cua ma tran A:")

# for i in A:
#     print(i[2])

# Bai 9
# print(np.random.randint(-10, 11, size=(3,3)))

# Bai 10
# Ma tran don vi:
# Duong cheo chinh = 1; con lai = 0
# identify_matrix = np.eye(3)
# print(identify_matrix)

# Bai 11
# print(np.random.randint(1, 11, size=(3,3)))

# row = 3
# col = 3
# A = []
# for r in range(row):
#     r = []
#     for c in range(col):
#         r.append(np.random.randint(1, 10))
#     A.append(r)
# print(A)

# Bai 12
# matrix_dcc = np.diag([1, 2, 3])
# print(matrix_dcc)

# Bai 13
# A = np.array([[1, 1, 2], [2, 4, -3], [3, 6, -5]])
# dinhThucA = np.linalg.det(A)
# print("Dinh thuc cua A: ", dinhThucA)

# Bai 14
# a1 = np.array([1, -2, -5])
# a2 = np.array([2, 5, 6])
# # column_stack: Xếp các chuỗi thành cột
# M = np.column_stack((a1, a2))
# print("Ma tran M M:\n", M)

# Bai 15
# 0.1: buoc nhay
x = np.arange(-5, 6, 0.1)
y = x**2

plt.plot(x, y)

plt.title("Do thi ham so y = x^2")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.show()

# Bai 16
x = np.linspace(-5, 5, 50)
y = x**2

# Vẽ đồ thị
plt.plot(x, y)
plt.title("Đồ thị hàm số y = x^2")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.show()

# Bai 18
x = np.linspace(-5, 5, 50)
y = np.exp(x)

plt.plot(x, y)
plt.title("Đồ thị hàm số y = exp(x)")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.show()

# Bai 19
x = np.linspace(0, 5, 50)
y = np.log(x)

plt.plot(x, y)
plt.title("Đồ thị hàm số y = log(x)")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.show()

# Bai 20
x = np.linspace(-5, 5, 100)
x_log = np.linspace(0, 5, 100)

y1 = np.exp(x)
y2 = np.exp(2 * x)

y3 = np.log(x_log)
y4 = np.log(2 * x_log)

# Do thi 1: 2 hang 1 cot, vi tri 1
plt.subplot(2, 1, 1)
plt.plot(x, y1, label="y = exp(x)")
plt.plot(x, y2, label="y = exp(2*x)")
plt.title("Đồ thị hàm số y = exp(x) và y = exp(2*x)")
plt.xlabel("x")
plt.ylabel("y")
# Hien chu thich
plt.legend()
plt.grid(True)

# Do thi 2: 2 hang 1 cot, vi tri 2
plt.subplot(2, 1, 2)
plt.plot(x_log, y3, label="y = log(x)")
plt.plot(x_log, y4, label="y = log(2*x)")
plt.title("Đồ thị hàm số y = log(x) và y = log(2*x)")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
# Dieu chinh khoang cach giua do thi, title, label
plt.tight_layout()

plt.show()
