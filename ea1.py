def mdd(theta,y,x,z):
    e=y-x@theta.reshape(33,2)
    meane=np.mean(e,0)
    e_demean=e-meane
    e_mat=e_demean.dot(e_demean.T)
    z=z.reshape(n,-1)
    z_mat=squareform(pdist(z,metric='euclidean'))
    return (abs(np.mean(e_mat*z_mat)))
def issmaller(a,b,n):
    return (a<=b).all(axis=1).reshape(n,1)
def DL(theta,y,x,z):
    e=y-x@theta.reshape(33,2)
    tot=0
    for l in range(n):
        tot+=np.sum(e*issmaller(z,z[l,:],n),axis=0).reshape(1,2)@np.sum(e*issmaller(z,z[l,:],n),axis=0).reshape(2,1)
    return(tot/n**3)
P=data["Close"]
Y=100*np.diff(np.log(P))
Y=Y.reshape(len(Y),-1)
r1 = 0
index=np.array(range(4,417))
n=len(index)
dim=1
IVlag=2
para_number=6
valid_samplesize=n  
convar=np.concatenate([Y[index-1],Y[index-2],Y[index-3],Y[index-4]],axis=1)  
theta_mddW=np.zeros([para_number,para_number])
theta_mdd=np.zeros([para_number,1])
theta_DLW=np.zeros([para_number,para_number])
theta_DL=np.zeros([para_number,1])
flag=(Y[index-1]<=r1)*1 
x1=flag.reshape(n,-1)
for l in range(2):
    x1=np.concatenate([x1,(Y[index-l-1]*flag).reshape(valid_samplesize,-1)],axis=1)
flag=(Y[index-1]>r1)*1
x3=flag.reshape(n,-1)
for l in range(2):
    x3=np.concatenate([x3,(Y[index-l-1]*flag).reshape(valid_samplesize,-1)],axis=1)
x=np.concatenate([x1,x3],axis=1)
y=Y[index]
theta_ols=np.linalg.inv(x.T@x)@x.T@y
for i in range(n):
    xi_demean=x[i,:]-np.mean(x,axis=0)
    for j in range(n):
        xj_demean=(x[j,:]-np.mean(x,axis=0)).reshape(1,6)
        yj_demean=(y[j]-np.mean(y)).reshape(1,1)
        theta_mddW+=np.kron(xi_demean,np.diag(np.ones(dim))).T@np.kron(xj_demean,np.diag(np.ones(dim)))*np.linalg.norm(convar[i,:]-convar[j,:])/n**2 
        theta_mdd+=np.kron(xi_demean,np.diag(np.ones(dim))).T@yj_demean*np.linalg.norm(convar[i,:]-convar[j,:])/n**2  
theta_mdd=np.linalg.pinv(theta_mddW)@theta_mdd
kernelDL=np.zeros([n,n]) 
for i in range(n):
    xi=x[i,:]
    for j in range(n):
        xj=x[j,:].reshape(1,6)
        yj=y[j].reshape(1,1)
        for k in range(n):
            kernelDL[i,j]+=all(convar[k,:]>=convar[i,:])*all(convar[k,:]>=convar[j,:])/n
        theta_DLW+=np.kron(xi,np.diag(np.ones(dim))).T@np.kron(xj,np.diag(np.ones(dim)))*kernelDL[i,j]/n**2  
        theta_DL+=np.kron(xi,np.diag(np.ones(dim))).T@yj*kernelDL[i,j]/n**2 
theta_DL=np.linalg.pinv(theta_DLW)@theta_DL
epsilon_mdd=y-x@(theta_mdd.reshape(6,1))
epsilon_DL=y-x@(theta_DL.reshape(6,1))
sum_mdd=np.zeros([n,dim,para_number])
dotHn=np.zeros([n,dim,para_number])
meandoth=np.zeros((dim,para_number))
for i in range(n): 
    latent=-np.kron(x[i,:],np.diag(np.ones(dim)))
    meandoth+=latent/n 
gam_DL=np.zeros([n,n,dim,dim])
for i in range(n):
    for j in range(n):
        doth=-np.kron(x[j,:],np.diag(np.ones(dim)))
        sum_mdd[i,:,:]+=(doth-meandoth)*np.linalg.norm(convar[i,:]-convar[j,:])/n  
        dotHn[i,:,:]+=doth*(all(convar[j,:]<=convar[i,:]))/n  
        for k in range(n):
            gam_DL[i,j,:,:]+=epsilon_DL[k].reshape(dim,1)@epsilon_DL[k].reshape(1,dim)*all(convar[k,:]<=convar[i,:])*all(convar[k,:]<=convar[j,:])/n  
Omega=np.zeros([para_number,para_number])
asd_mdd=np.zeros([para_number,para_number])
sum_mdd_mean=np.mean(sum_mdd,axis=0)  
for i in range(n):
    doth=-np.kron(x[i,:],np.diag(np.ones(dim)))
    Omega+=(doth-meandoth).T@sum_mdd[i,:,:]/n
    asd_mdd+=(sum_mdd[i,:,:]-sum_mdd_mean).T@epsilon_mdd[i].reshape(dim,1)@epsilon_mdd[i].reshape(1,dim)@(sum_mdd[i,:,:]-sum_mdd_mean)/n
asd_mdd=np.linalg.pinv(Omega)@asd_mdd@np.linalg.pinv(Omega)
asd_DL=np.zeros([para_number,para_number])
W_DL=np.zeros([para_number,para_number])
for sb in range(n):
    W_DL+=dotHn[sb,:,:].T@dotHn[sb,:,:]
    for sb2 in range(n):
        asd_DL+=dotHn[sb,:,:].T@gam_DL[sb,sb2,:]@dotHn[sb2,:,:]
asd_DL=np.linalg.pinv(W_DL)@asd_DL@np.linalg.pinv(W_DL)
print(theta_mdd)
print(theta_DL)
print(np.diag(np.sqrt(asd_mdd/n)))
print(np.diag(np.sqrt(asd_DL/n)))