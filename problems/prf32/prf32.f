      program PRF32
     
c    TESTING OF THE PREFETCH DIRECTIVE. 

      print *,'===START OF PRF32========================'
C --------------------------------------------------
      call prf3201
      call prf3202
      call prf3203
C
      print *,'=== END OF PRF32 ========================= '    
      end
C ---------------------------------------------------------PRF3201
      subroutine PRF3201
      integer, parameter ::  N = 16,M=8,K=8,NL=1000,NIT=3
      integer, allocatable :: A(:,:,:),B(:,:,:),C(:,:,:),A1(:,:,:)
      integer nloopi,nloopj,nloopii 
      character*7 tname
                 
cdvm$ distribute B(BLOCK,BLOCK,*)
cdvm$ align(:,:,:) with B(:,:,:) :: A,A1

cdvm$ remote_group GR1
cdvm$ remote_group GR2
cdvm$ remote_group GR3 

      tname='PRF3201'
      allocate (B(N,M,K),A(N,M,K),C(N,M,K),A1(N,M,K))
      NNL=NL    
      call serial3(C,N,M,K,NNL)
      nloopi=NL
      nloopj=NL
      nloopii=NL
*dvm$ parallel (i,j,ii) on A(i,j,ii)
      do i=1,N
        do j=1,M
          do ii=1,K
            A(i,j,ii) = NL+i+j+ii
          enddo
        enddo                                         
      enddo                                        
      do it=1,NIT
cdvm$ prefetch GR1                                                  
cdvm$ prefetch GR2 
cdvm$ prefetch GR3 
               
cdvm$ remote_access (GR1:A(1,1,1))
      ib1=A(1,1,1)               

cdvm$ remote_access (GR1:A(N,M,K))
      ib2=A(N,M,K)               

cdvm$ remote_access (GR2:A(1,M,K))
      ib3=A(1,M,K)               

cdvm$ remote_access (GR3:A(N,1,K))
      ib4=A(N,1,K)               

cdvm$ remote_access (GR3:A(N,M,1))
      ib5=A(N,M,1)               

      if ((ib1 .eq.C(1,1,1)) .and.(ib2 .eq.C(N,M,K)) .and.
     * (ib3 .eq.C(1,M,K)) .and.(ib4 .eq.C(N,1,K)) .and.
     * (ib5 .eq.C(N,M,1))) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      if (it .eq. 2) cycle 
cdvm$ reset GR1
cdvm$ reset GR2
cdvm$ reset GR3

      enddo
      deallocate (A,B,C,A1)
      end

C ------------------------------------------------------PRF3202
      subroutine PRF3202
      integer, parameter ::  N = 16,M=8,K=8,NL=1000,NIT=3
      integer, allocatable :: A(:,:,:),B(:,:,:),C(:,:,:),D(:,:,:)
      integer nloopi,nloopj,nloopii 
      character*7 tname
                 
cdvm$ distribute A(BLOCK,*,BLOCK)
cdvm$ align(:,:,:) with A(:,:,:) :: B 

cdvm$ remote_group GR1
cdvm$ remote_group GR2
cdvm$ remote_group GR3

      tname='PRF3202'
      allocate (A(N,M,K),B(N,M,K),C(N,M,K),D(N,M,K))
      NNL=NL    
      call serial3(C,N,M,K,NNL)
      nloopi=NL
      nloopj=NL
      nloopii=NL
*dvm$ parallel (i,j,ii) on A(i,j,ii)
      do i=1,N
        do j=1,M
          do ii=1,K
            A(i,j,ii) = NL+i+j+ii
          enddo
        enddo                                         
      enddo                                        
      do it=1,NIT
cdvm$ prefetch GR1                                                  
cdvm$ prefetch GR2 
cdvm$ prefetch GR3 
               
      isumc1=0
      isuma1=0

cdvm$ remote_access (GR1:A(:,:,:))               
      do i=1,N         
       do j=i,M
        do ii=1,K
         D(i,j,ii)=A(i,j,ii)
         isumc1=isumc1+C(i,j,ii)
         isuma1=isuma1+D(i,j,ii)
        enddo
       enddo
      enddo

      isumc2=0
      isuma2=0

cdvm$ remote_access (GR1:A(1,:,:))
      do j=1,M         
       do ii=1,K
        D(1,j,ii)=A(1,j,ii)
        isumc2=isumc2+C(1,j,ii)
        isuma2=isuma2+D(1,j,ii)
       enddo
      enddo

      isumc3=0
      isuma3=0

cdvm$ remote_access (GR1:A(:,M,:))
      do i=1,N         
       do ii=1,K
        D(i,M,ii)=A(i,M,ii)
        isumc3=isumc3+C(i,M,ii)
        isuma3=isuma3+D(i,M,ii)
       enddo
      enddo 

      isumc4=0
      isuma4=0

cdvm$ remote_access (GR2:A(:,:,K))
      do i=1,N
       do j=1,M         
        D(i,j,K)=A(i,j,K)
        isumc4=isumc4+C(i,j,K)
        isuma4=isuma4+D(i,j,K)
       enddo
      enddo 
 
      ki=2
      ki1=3
      kj=2
      kj1=3
      kii=2
      kii1=3
 
      isumc5=0
      isuma5=0       
cdvm$ remote_access (GR3:A(:,:,:))
      do i=1,N/ki-ki1         
       do j=1,M/kj-kj1
        do ii=1,K/kii-kii1
         D(i,j,ii)=A(ki*i+ki1,kj*j+kj1,kii*ii+kii1)
         isumc5=isumc5+C(ki*i+ki1,kj*j+kj1,kii*ii+kii1)
         isuma5=isuma5+D(i,j,ii)
        enddo
       enddo
      enddo
      if ((isumc1 .eq.isuma1).and.(isumc2 .eq.isuma2).and. 
     *      (isumc3 .eq.isuma3).and.(isumc4 .eq.isuma4).and.
     *      (isumc5 .eq.isuma5)) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      if (it .eq. 2) cycle 
cdvm$ reset GR1
cdvm$ reset GR2
cdvm$ reset GR3

      enddo  
      deallocate (A,B,C,D)
     
      end
C ------------------------------------------------------PRF3203
      subroutine PRF3203
      integer, parameter ::  N = 16,M=8,K=8,NL=1000,NIT=3
      integer, allocatable :: A(:,:,:),B(:,:,:),C(:,:,:),A1(:,:,:)
      integer nloopi,nloopj,nloopii 
      character*7 tname
                 
cdvm$ distribute A(*,BLOCK,BLOCK)
cdvm$ align(:,:,:) with A(:,:,:) :: B ,A1

cdvm$ remote_group GR1
cdvm$ remote_group GR2
cdvm$ remote_group GR3

      tname='PRF3202'
      allocate (A(N,M,K),B(N,M,K),C(N,M,K),A1(N,M,K))
      NNL=NL    
      call serial3(C,N,M,K,NNL)
*dvm$ parallel (i,j,ii) on A(i,j,ii)
      do i=1,N
        do j=1,M
          do ii=1,K
           A(i,j,ii) = NL+i+j+ii
           A1(i,j,ii) = NL+i+j+ii
          enddo
        enddo                                         
      enddo
                                         
      do it=1,NIT
cdvm$ prefetch GR1                                                  
cdvm$ prefetch GR2 
cdvm$ prefetch GR3 

      nloopi1=NL
      nloopj1=NL
      nloopii1=NL
               
*dvm$ parallel (i,j,ii) on B(i,j,ii),remote_access(GR1:A(1,1,1))
      do i=1,N
       do j=1,M
         do ii=1,K
          B(i,j,ii) = A(1,1,1)
         enddo
       enddo
      enddo

*dvm$ parallel (i,j,ii) on B(i,j,ii),
*dvm$* reduction( min( nloopi1),min(nloopj1),min(nloopii1))
      do i=1,N
       do j=1,M
        do ii=1,K
          if (B(i,j,ii).ne.C(1,1,1)) then
           nloopi1=min(nloopi1,i)
           nloopj1=min(nloopj1,j)
           nloopii1=min(nloopii1,ii)
          endif
        enddo
       enddo
      enddo 

      nloopi2=NL
      nloopj2=NL
      nloopii2=NL

*dvm$ parallel (i,j,ii) on B(i,j,ii),remote_access(GR1:A(N,M,K))
      do i=1,N
       do j=1,M
        do ii=1,K
         B(i,j,ii) = A(N,M,K)
        enddo
       enddo
      enddo 

*dvm$ parallel (i,j,ii) on B(i,j,ii),
*dvm$*reduction( min( nloopi2),min(nloopj2),min(nloopii2))
      do i=1,N
       do j=1,M
         do ii=1,K
          if (B(i,j,ii).ne.C(N,M,K)) then
           nloopi2=min(nloopi2,i)
           nloopj2=min(nloopj2,j)
           nloopii2=min(nloopii2,ii)
          endif
         enddo
       enddo
      enddo

      nloopi3=NL
      nloopj3=NL
      nloopii3=NL

*dvm$ parallel (i,j,ii) on B(i,j,ii),remote_access(GR2:A)
      do i=1,N
       do j=1,M
        do ii=1,K
         B(i,j,ii) = A(i,j,ii)      
        enddo
       enddo
      enddo 

*dvm$ parallel (i,j,ii) on B(i,j,ii),
*dvm$* reduction( min( nloopi3),min(nloopj3),min(nloopii3))
       do i=1,N
        do j=1,M
         do ii=1,K
          if (B(i,j,ii).ne.C(i,j,ii)) then
           nloopi3=min(nloopi3,i)
           nloopj3=min(nloopj3,j)
           nloopii3=min(nloopii3,ii)
          endif
        enddo
       enddo
      enddo                                     
   
      nloopi4=NL
      nloopj4=NL
      nloopii4=NL            

*dvm$ parallel (i,j,ii) on B(i,j,ii),remote_access(GR2:A(1,1,1))
      do i=1,N
       do j=1,M
        do ii=1,K
         B(i,j,ii) = A(1,1,1)
        enddo
       enddo
      enddo

*dvm$ parallel (i,j,ii) on B(i,j,ii),
*dvm$* reduction( min( nloopi4),min(nloopj4),min(nloopii4))
      do i=1,N
       do j=1,M
        do ii=1,K
          if (B(i,j,ii).ne.C(1,1,1)) then
           nloopi4=min(nloopi4,i)
           nloopj4=min(nloopj4,j)
           nloopii4=min(nloopii4,ii)
          endif
         enddo
        enddo
       enddo

      nloopi5=NL
      nloopj5=NL
      nloopii5=NL

*dvm$ parallel (i,j,ii) on A(i,j,ii),remote_access(GR3:A(1,:,:))
      do i=1,N
       do j=1,M
        do ii=1,K
         B(i,j,ii) = A(1,j,ii)      
        enddo
       enddo
      enddo 

*dvm$ parallel (i,j,ii) on B(i,j,ii),
*dvm$*reduction( min( nloopi5),min(nloopj5),min(nloopii5))
       do i=1,N
        do j=1,M
         do ii=1,K
          if (B(i,j,ii).ne.C(1,j,ii)) then
           nloopi5=min(nloopi5,i)
           nloopj5=min(nloopj5,j)
           nloopii5=min(nloopii5,ii)
          endif
         enddo
        enddo
       enddo

      nloopi6=NL
      nloopj6=NL
      nloopii6=NL

*dvm$ parallel (i,j,ii) on B(i,j,ii),remote_access(GR3:A1(:,M,:))
      do i=1,N
       do j=1,M
        do ii=1,K
         B(i,j,ii) = A1(i,M,ii)      
        enddo
       enddo
      enddo
 
*dvm$ parallel (i,j,ii) on B(i,j,ii),
*dvm$* reduction( min( nloopi6),min(nloopj6),min(nloopii6))
      do i=1,N
       do j=1,M
        do ii=1,K
         if (B(i,j,ii).ne.C(i,M,ii)) then
           nloopi6=min(nloopi6,i)
           nloopj6=min(nloopj6,j)
           nloopii6=min(nloopii6,ii)
         endif
        enddo
       enddo
      enddo

      nloopi7=NL
      nloopj7=NL
      nloopii7=NL

*dvm$ parallel (i,j,ii) on A(i,j,ii),
*dvm$*remote_access(GR3:A1(:,:,K))
      do i=1,N
       do j=1,M
        do ii=1,K
         B(i,j,ii) = A1(i,j,K)      
        enddo
       enddo
      enddo
 
*dvm$ parallel (i,j,ii) on A(i,j,ii),
*dvm$*reduction( min( nloopi7),min(nloopj7),min(nloopii7))
      do i=1,N
       do j=1,M
        do ii=1,K
         if (B(i,j,ii).ne.C(i,j,K)) then
           nloopi7=min(nloopi7,i)
           nloopj7=min(nloopj7,j)
           nloopii7=min(nloopii7,ii)
         endif
        enddo
       enddo
      enddo
      if ((nloopi1 .eq.NL).and.(nloopi2 .eq.NL).and.
     *     (nloopi3 .eq.NL).and. (nloopi4 .eq.NL).and.
     *     (nloopi5 .eq.NL).and.(nloopi6 .eq.NL).and.
     *     (nloopi7 .eq.NL)) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
   
      if (it .eq. 2) cycle 
cdvm$ reset GR1
cdvm$ reset GR2
cdvm$ reset GR3
      enddo    
      deallocate (A,B,C,A1)
      end



C ---------------------------------------------------------         
      subroutine serial3(AR,N,M,K,NL)
      integer AR(N,M,K)
      integer NL 
      do 10 i=1,N
      do 10 j=1,M
      do 10 ii=1,K
   10 AR(i,j,ii) = NL+i+j+ii
                   
      end 


      subroutine ansyes(name)
      character*7 name
      print *,name,'  -  complete'
      end
      subroutine ansno(name)
      character*7 name
      print *,name,'  -  ***error'
      end
