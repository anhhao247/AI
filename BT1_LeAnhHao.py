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

#Bai 4
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
# sorted_result = sorted(result, key=lambda x: x[0])
# # print(sorted_result)
# for a, n in sorted_result:
#     print(a, ": ", n)

# Bai 9
# File
