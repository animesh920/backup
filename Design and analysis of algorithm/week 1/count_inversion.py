def _SortAndCountInversions(lst, start, end):
    if (start == end):
        return 0
    else:
        x = _SortAndCountInversions(lst, start, (end+start)//2)
        y = _SortAndCountInversions(lst, (end+start)//2+1, end)
        z = _MergeAndCountSplitInversions(lst, start, (end+start)//2+1, end)
        return (x+y+z)


def _MergeAndCountSplitInversions(lst, start, m, end):
    i = start
    j = m
    res = []
    
    inv = 0
    
    while (i < m or j <= end):
        if (i == m):
            res.append(lst[j])
            j += 1
        elif (j > end):
            res.append(lst[i])
            i += 1
        elif (lst[i] <= lst[j]):
            res.append(lst[i])
            i += 1
        else:
            inv += (m-i) 
            res.append(lst[j])
            j += 1
        
    i = start
    while (i <= end):
        lst[i] = res[i-start]
        i += 1

    return inv


def CountInversions(lst):
    return _SortAndCountInversions(lst, 0, len(lst)-1)
    
    
def main():
    # Test Cases
    print(CountInversions([1,2,3,4,5,6]))
    print(CountInversions([6,5,4,3,2,1]))
    print(CountInversions([1,3,5,2,4,6]))
    
    # Assignment data
    f = open('IntegerArray.txt', 'r')
    lst = []
    line = f.readline()
    while (line != ''):
        lst.append(int(line))
        line = f.readline()
    
    print(CountInversions(lst))

if __name__ == '__main__':
    main()