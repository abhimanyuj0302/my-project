li = [1,2,3,4,5,6,7,8,9,10]
'''
evenli = []

for var in li:
    if var %2 ==0:
        evenli.append(var)

print("list even :", evenli)
'''

# list comprehention for even 

evenli = [var for var in li if var %2 == 0]
print(evenli)

'''
oddli = []
for var in li :
    if var %2 ==1 :
        oddli.append(var)

print("list odd :",oddli)
'''
# list comprehention for even 

oddli = [var for var in li if var %2 == 1]
print(oddli)

'''
sqr = []
for var in li:
    s = var * var
    sqr.append(s)

print(sqr)
'''

# list comprehention for even 

sqr = [var*var for var in li]
print(sqr)

