
def mdd(theta,y,x,z): 
    e=y-(x@theta.reshape(2,2))
    meane=np.mean(e,0) 
    e_demean=e-meane 
    e_mat=e_demean.dot(e_demean.T) 
    z=z.reshape(n,-1)
    z_mat=squareform(pdist(z,metric='euclidean')) 
    return (abs(np.mean(e_mat*z_mat))) 
def issmaller(a,b):
    return (a<=b).all(axis=1).reshape(n,1)
def DL(theta,y,x,z):
    e=y-(x@theta.reshape(2,2))
    tot=0
    for l in range(n):
        tot+=np.sum(e*issmaller(z,z[l,:]),axis=0).reshape(1,2)@np.sum(e*issmaller(z,z[l,:]),axis=0).reshape(2,1)
    return(tot/n**3)
phi1=0.1
phi2=0.1
phi5=0.1
phi6=0.1
phi3=0.8
phi7=0.7
phi4=0.8
A=np.array([1,-1,1,2]).reshape(2,2)
for n in [50,100,200]:
    monte0=1000
    def process(monte):
        theta_mdd=np.zeros(4)
        theta_DL=np.zeros(4)
        asd_mdd=np.zeros((4,4))
        asd_DL=np.zeros((4,4))
        time.sleep(0.1)
        np.random.seed(monte)
        h=np.random.multivariate_normal([0,0],np.diag([1,1]),n)
        epsilon=np.zeros([n,2])
        V=np.zeros((n,2,2))
        y=np.zeros((n,2))
        V[0,0,0]=phi1
        V[0,1,1]=phi2
        V[0,0,1]=phi7*np.sqrt(phi1*phi2)
        V[0,1,0]=phi7*np.sqrt(phi1*phi2)
        epsilon[0,:]=sqrtm(V[0,:,:])@eta[0,:]
        for ind in range(n-1):
            V[ind+1,0,0]=phi1+phi3*V[ind,0,0]+phi5*epsilon[ind,0]**2
            V[ind+1,1,1]=phi2+phi4*V[ind,1,1]+phi6*epsilon[ind,1]**2
            V[ind+1,0,1]=phi7*np.sqrt(V[ind+1,0,0]*V[ind+1,1,1])
            V[ind+1,1,0]=phi7*np.sqrt(V[ind+1,0,0]*V[ind+1,1,1])
            epsilon[ind+1,:]=sqrtm(V[ind+1,:,:])@eta[ind+1,:]
        x=np.zeros((n,2))
        zeta=np.random.multivariate_normal([0,0],np.diag([1,1]),n)
        x[0,:]=zeta[0,:]
        for ind in range(n-1):
            x[ind+1,:]=np.diag([0.3,0.2])@x[ind,:]+zeta[ind+1,:]
        y=x@A+epsilon
        theta_mdd=minimize(fun=lambda theta: mdd(theta,y=y,x=x,z=x),x0=A.flatten()).x
        theta_DL=minimize(fun=lambda theta: DL(theta,y=y,x=x,z=x),x0=A.flatten()).x
        epsilon_mdd=y-x@theta_mdd.reshape(2,2)
        epsilon_DL=y-x@theta_DL.reshape(2,2)
        sum_mdd=np.zeros([n,2,4])
        dotHn=np.zeros([n,2,4])
        meandoth_beforeswap=np.zeros((2,4))
        meandoth=np.zeros((2,4))
        for i in range(n):
            meandoth_beforeswap+=-np.kron(x[i,:],np.diag([1,1]))/n
        meandoth=meandoth_beforeswap
        Omega=np.zeros([4,4])
        gam_DL=np.zeros([n,n,2,2])
        for i in range(n):
            for j in range(n):
                doth_beforeswap=-np.kron(x[j,:],np.diag([1,1])) 
                doth=np.zeros_like(doth_beforeswap) 
                doth=doth_beforeswap
                sum_mdd[i,:]+=(doth-meandoth)*np.linalg.norm(x[i,:]-x[j,:])/n  
                dotHn[i,:]+=doth*(all(x[j,:]<=x[i,:]))/n 
                for k in range(n):
                    gam_DL[i,j,:]+=epsilon_DL[k,:].reshape(2,1)@epsilon_DL[k,:].reshape(1,2)*all(x[k,:]<=x[i,:])*all(x[k,:]<=x[j,:])/n #Gam(Xi,Xj)
        sum_mdd_mean=np.mean(sum_mdd,0)  
        for i in range(n):
            doth_beforeswap=-np.kron(x[i,:],np.diag([1,1])) 
            doth=doth_beforeswap
            Omega+=(doth-meandoth).T@sum_mdd[i,:]/n   
            asd_mdd+=(sum_mdd[i,:]-sum_mdd_mean).T@epsilon_mdd[i,:].reshape(2,1)@epsilon_mdd[i,:].reshape(1,2)@(sum_mdd[i,:]-sum_mdd_mean)/n
        asd_mdd=np.linalg.inv(Omega)@asd_mdd@np.linalg.inv(Omega)
        W_DL=np.zeros([4,4])
        for k in range(n):
            W_DL+=dotHn[k,:].T@dotHn[k,:]
            for k2 in range(n):
                asd_DL+=dotHn[k,:].T@gam_DL[k,k2,:]@dotHn[k2,:]
        asd_DL=np.linalg.inv(W_DL)@asd_DL@np.linalg.inv(W_DL)
        asd_mddd=np.diag(asd_mdd/n)
        asd_DLd=np.diag(asd_DL/n)
        return(theta_mdd,theta_DL,np.sqrt(asd_mddd),np.sqrt(asd_DLd))
    with Pool(24) as pool:
        results=np.array(pool.map(process,range(monte0)))
    print(np.mean(results[:,0,:],axis=0)-A.flatten())
    print(np.mean(results[:,1,:],axis=0)-A.flatten())  
    print(np.mean(results[:,2,:],axis=0))
    print(np.mean(results[:,3,:],axis=0))
    print(np.sqrt(np.diag(np.cov(results[:,0,:].T))))
    print(np.sqrt(np.diag(np.cov(results[:,1,:].T))))