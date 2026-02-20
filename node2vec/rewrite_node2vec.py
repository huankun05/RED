def rewrite_node2vec(args,outpathbasis='matrix_node2vec'):
    filein ='embedding_node2vec' + str(args.q) + '.txt'
    #filein = 'testin.txt'
    fileout = outpathbasis+str(args.q)+'.txt'

    f = open(filein, 'r')
    of = open(fileout, 'w')

    i = 0
    f_lines = f.readlines()
    for line in f_lines:
        if i==0:
            line = line.strip()
            vec = line.split()
            N,d=int(vec[0]),int(vec[1])
            matrix=[]
            for k in range(N):
                matrix.append([0]*d)
        else:
            line = line.strip()
            vec = line.split()
            row=int(vec[0])-1  #the node id should start at 1 instead of 0
            for j in range(1,d+1):
                matrix[row][j-1]=vec[j] #type=str
            #print(row,matrix)
        i = (i + 1)
    j=0
    for mline in matrix:
        wline=''
        for item in mline:
            wline=wline+item+' '
        j=j+1
        if j<N:
            wline = wline+ '\n'
        of.write(wline)
    f.close()
    of.close()
    if i!=(N+1):
        print('Error: line error!')
        print('N=',N)
        print('lines=',i-1)

#rewrite_node2vec()
