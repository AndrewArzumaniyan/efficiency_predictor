      program PRF22
     
c    TESTING OF THE PREFETCH DIRECTIVE. 

      print *,'===START OF PRF22========================'
C --------------------------------------------------
      call prf2201
      call prf2202
      call prf2203
C
      print *,'=== END OF PRF22 ========================= '    
      end
C ---------------------------------------------------------PRF2201
      subroutine PRF2201
      integer, parameter ::  N = 4,M=4,NL=1000,NIT=3
      integer, allocatable :: A(:,:),B(:,:),C(:,:),D(:,:)
      integer,allocatable :: A1(:,:),A2(:,:),A3(:,:)
      integer nloopi,nloopj 
      character*7 tname
                 
cdvm$ distribute B(BLOCK,*)    
cdvm$ align(:,:) with B(:,:) :: A,A1,A2,A3
 
cdvm$ remote_group GR1
cdvm$ remote_group GR2
cdvm$ remote_group GR3

      tname='PRF2201'
      allocate (B(N,M),A(N,M),C(N,M),D(N,M))
      allocate (A1(N,M),A2(N,M),A3(N,M))
      NNL=NL    
      call serial2(C,N,M,NNL)
      nloopi=NL
      nloopj=NL
*dvm$ parallel (i,j) on A(i,j)
      do i=1,N
        do j=1,M
         A(i,j) = NL+i+j
      enddo
      enddo                                         
      do it=1,NIT
cdvm$ prefetch GR1                                                  
cdvm$ prefetch GR2 
cdvm$ prefetch GR3 

cdvm$ remote_access (GR1:A(1,1))
      ib1=A(1,1)

cdvm$ remote_access (GR1:A(N,M))
      ib2=A(N,M)            

cdvm$ remote_access (GR2:A(1,M))
      ib3=A(1,M)            

cdvm$ remote_access (GR3:A(N,1))
      ib4=A(N,1)

      if ((ib1 .eq.C(1,1)).and.(ib2.eq.C(N,M)).and.(ib3.eq.C(1,M)).and.
     *      (ib4 .eq. C(N,1)) ) then     
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
      deallocate (A1,A2,A3)

      end 
C ---------------------------------------------------------PRF2202
      subroutine PRF2202
      integer, parameter ::  N = 4,M=4,NL=1000,NIT=3
      integer, allocatable :: A(:,:),B(:,:),C(:,:),D(:,:)
      integer,allocatable :: A1(:,:),A2(:,:),A3(:,:)
      integer nloopi,nloopj 
      character*7 tname
                 
cdvm$ distribute B(*,BLOCK)    
cdvm$ align(:,:) with B(:,:) :: A,A1,A2,A3
 
cdvm$ remote_group GR1
cdvm$ remote_group GR2
cdvm$ remote_group GR3

      tname='PRF2202'
      allocate (B(N,M),A(N,M),C(N,M),D(N,M))
      allocate (A1(N,M),A2(N,M),A3(N,M))
      NNL=NL    
      call serial2(C,N,M,NNL)
      nloopi=NL
      nloopj=NL
*dvm$ parallel (i,j) on A(i,j)
      do i=1,N
        do j=1,M
         A(i,j) = NL+i+j
         A1(i,j) =NL+i+j
         A2(i,j) =NL+i+j
        enddo
      enddo                                         
      do it=1,NIT
cdvm$ prefetch GR1                                                  
cdvm$ prefetch GR2 
cdvm$ prefetch GR3 
               
      isumc1=0
      isuma1=0

cdvm$ remote_access (GR1:A(:,:))
      do i=1,N         
      do j=i,M
       D(i,j)=A(i,j)
       isumc1=isumc1+C(i,j)
       isuma1=isuma1+D(i,j)
      enddo
      enddo

      isumc2=0
      isuma2=0
      
cdvm$ remote_access (GR1:A(:,1))
      do i=1,N               
       D(i,1)=A(i,1)
       isumc2=isumc2+C(i,1)
       isuma2=isuma2+D(i,1)
      enddo

      isumc3=0
      isuma3=0

cdvm$ remote_access (GR2:A(1,:))
      do j=1,M         
       D(1,j)=A(1,j)
       isumc3=isumc3+C(1,j)
       isuma3=isuma3+D(1,j)
      enddo 

      isumc4=0
      isuma4=0

cdvm$ remote_access (GR2:A(:,M))
      do i=1,N         
       D(i,M)=A(i,M)
       isumc4=isumc4+C(i,M)
       isuma4=isuma4+D(i,M)
      enddo

      isumc5=0
      isuma5=0
cdvm$ remote_access (GR2:A1(N,:))
      do j=1,M         
       D(N,j)=A1(N,j)
       isumc5=isumc5+C(N,j)
       isuma5=isuma5+D(N,j)
      enddo

      isumc6=0
      isuma6=0                                                                 

cdvm$ remote_access (GR3:A1(:,:))
      do i=1,N         
       do j=i,M
        D(i,j)=A1(i,j)
        isumc6=isumc6+C(i,j)
        isuma6=isuma6+D(i,j)
       enddo
      enddo

      isumc7=0
      isuma7=0

      ki=2
      ki1=3
      kj=2
      kj1=3

cdvm$ remote_access (GR3:A2(:,:))
      do i=1,N/ki-ki1         
       do j=i,M/kj-kj1
        D(i,j)=A2(ki*i+ki1,kj*j+kj1)
        isumc7=isumc7+C(ki*i+ki1,kj*j+kj1 )
        isuma7=isuma7+D(i,j)
       enddo
      enddo

      if ((isumc1.eq.isuma1).and.(isumc2 .eq.isuma2).and.
     * (isumc3.eq.isuma3)
     * .and.(isumc4 .eq.isuma4).and.(isumc5 .eq.isuma5).and.
     * (isumc6 .eq.isuma6).and.(isumc7 .eq.isuma7)) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif
c      print *,isumc1,isuma1,isumc2,isuma2 
            
      if (it .eq. 2) cycle 
cdvm$ reset GR1
cdvm$ reset GR2
cdvm$ reset GR3

      enddo
      deallocate (A,B,C,D)
      deallocate (A1,A2,A3)

      end
C ---------------------------------------------------------PRF2203
      subroutine PRF2203
      integer, parameter ::  N = 4,M=4,NL=1000,NIT=3
      integer, allocatable :: A(:,:),B(:,:),C(:,:),D(:,:)
      integer,allocatable :: A1(:,:),A2(:,:),A3(:,:)
      integer nloopi,nloopj 
      character*7 tname
                 
cdvm$ distribute B(BLOCK,*)    
cdvm$ align(:,:) with B(:,:) :: A,A1,A2,A3
 
cdvm$ remote_group GR1
cdvm$ remote_group GR2
cdvm$ remote_group GR3

      tname='PRF2203'
      allocate (B(N,M),A(N,M),C(N,M),D(N,M))
      allocate (A1(N,M),A2(N,M),A3(N,M))
      NNL=NL    
      call serial2(C,N,M,NNL)
      nloopi=NL
      nloopj=NL
*dvm$ parallel (i,j) on A(i,j)
      do i=1,N
        do j=1,M
         A(i,j) = NL+i+j
         A1(i,j) = NL+i+j
        enddo
      enddo                                         
      do it=1,NIT
cdvm$ prefetch GR1                                                  
cdvm$ prefetch GR2 
cdvm$ prefetch GR3 
                             
      nloopi1=NL
      nloopj1=NL

*dvm$ parallel (i,J) on B(i,j),remote_access(GR1:A(1,1))
      do i=1,N
       do j=1,M
         B(i,j) = A(1,1)
       enddo
      enddo

*dvm$ parallel (i,j) on B(i,j), reduction( min( nloopi1),min(nloopj1))
      do i=1,N
       do j=1,M
          if (B(i,j).ne.C(1,1)) then
           nloopi1=min(nloopi1,i)
           nloopj1=min(nloopj1,j)
          endif
       enddo
      enddo

      nloopi2=NL
      nloopj2=NL

*dvm$ parallel (i,J) on B(i,j),remote_access(GR1:A(N,M))
      do i=1,N
       do j=1,M
         B(i,j) = A(N,M)
       enddo
      enddo 

*dvm$ parallel (i,j) on B(i,j), reduction( min( nloopi2),min(nloopj2))
      do i=1,N
       do j=1,M
          if (B(i,j).ne.C(N,M)) then
           nloopi2=min(nloopi2,i)
           nloopj2=min(nloopj2,j)
          endif
       enddo
      enddo

      nloopi3=NL
      nloopj3=NL

*dvm$ parallel (i,J) on B(i,j),remote_access(GR2:A(1,M))
      do i=1,N
       do j=1,M
         B(i,j) = A(1,M)
       enddo
      enddo 

*dvm$ parallel (i,j) on B(i,j), reduction( min( nloopi3),min(nloopj3))
      do i=1,N
       do j=1,M
          if (B(i,j).ne.C(1,M)) then
           nloopi3=min(nloopi3,i)
           nloopj3=min(nloopj3,j)
          endif
       enddo
      enddo

      nloopi4=NL
      nloopj4=NL

*dvm$ parallel (i,J) on B(i,j),remote_access(GR2:A(N,1))
      do i=1,N
       do j=1,M
         B(i,j) = A(N,1)
       enddo
      enddo 

*dvm$ parallel (i,j) on B(i,j), reduction( min( nloopi4),min(nloopj4))
      do i=1,N
       do j=1,M
          if (B(i,j).ne.C(N,1)) then
           nloopi4=min(nloopi4,i)
           nloopj4=min(nloopj4,j)
          endif
       enddo
      enddo 

      nloopi5=NL
      nloopj5=NL

*dvm$ parallel (i,J) on A(i,j),remote_access(GR3:A)
c *dvm$ parallel (i,J) on B(i,j),remote_access(A(:,:))
      do i=1,N
       do j=1,M
         B(i,j) = A(i,j)
       enddo
      enddo 

*dvm$ parallel (i,j) on B(i,j), reduction( min( nloopi5),min(nloopj5))
      do i=1,N
       do j=1,M
          if (B(i,j).ne.C(i,j)) then
           nloopi5=min(nloopi5,i)
           nloopj5=min(nloopj5,j)
          endif
       enddo
      enddo 

      nloopi6=NL
      nloopj6=NL

*dvm$ parallel (i) on B(i,1),remote_access(GR3:A1(:,1))
      do i=1,N
         B(i,1) = A1(i,1)
      enddo
      
*dvm$ parallel (i) on B(i,1), reduction( min( nloopi6),min(nloopj6))
      do i=1,N
          if (B(i,1).ne.C(i,1)) then
           nloopi6=min(nloopi6,i)
           nloopj6=min(nloopj6,j)
          endif
      enddo

      nloopi7=NL
      nloopj7=NL

*dvm$ parallel (i,J) on A(i,j),remote_access(GR3:A1(1,:))
      do i=1,N
       do j=1,M
         B(i,j) = A1(1,j)
       enddo
      enddo 

*dvm$ parallel (i,j) on B(i,j), reduction( min( nloopi7),min(nloopj7))
      do i=1,N
       do j=1,M
          if (B(i,j).ne.C(1,j)) then
           nloopi7=min(nloopi7,i)
           nloopj7=min(nloopj7,j)
          endif
       enddo
      enddo 

      nloopi8=NL
      nloopj8=NL

*dvm$ parallel (i,J) on B(i,j),remote_access(GR3:A1(:,M))
      do i=1,N
       do j=1,M
         B(i,j) = A1(i,M)
       enddo
      enddo 

*dvm$ parallel (i,j) on B(i,j), reduction( min( nloopi8),min(nloopj8))
      do i=1,N
       do j=1,M
          if (B(i,j).ne.C(i,M)) then
           nloopi8=min(nloopi8,i)
           nloopj8=min(nloopj8,j)
          endif
       enddo
      enddo

      nloopi9=NL
      nloopj9=NL

*dvm$ parallel (i,J) on A(i,j),remote_access(GR3:A1(N,:))
      do i=1,N
       do j=1,M
         B(i,j) = A1(N,j)
       enddo
      enddo 

*dvm$ parallel (i,j) on A(i,j), reduction( min( nloopi9),min(nloopj9))
      do i=1,N
       do j=1,M
          if (B(i,j).ne.C(N,j)) then
           nloopi9=min(nloopi9,i)
           nloopj9=min(nloopj9,j)
          endif
       enddo
      enddo         
      if ((nloopi1 .eq.NL).and.(nloopj1 .eq.NL) .and.
     *      (nloopi2 .eq.NL).and.(nloopj2 .eq.NL) .and.
     *      (nloopi3 .eq.NL).and.(nloopj3 .eq.NL) .and.
     *      (nloopi4 .eq.NL).and.(nloopj4 .eq.NL) .and.
     *      (nloopi5 .eq.NL).and.(nloopj5 .eq.NL) .and.
     *      (nloopi6 .eq.NL).and.(nloopj6 .eq.NL) .and.
     *      (nloopi7 .eq.NL).and.(nloopj7 .eq.NL) .and.
     *      (nloopi8 .eq.NL).and.(nloopj8 .eq.NL) .and.
     *      (nloopi9 .eq.NL).and.(nloopj9 .eq.NL)) then     
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
      deallocate (A1,A2,A3)

      end

C ---------------------------------------------------------         
      subroutine serial2(AR,N,M,NL)
      integer AR(N,M)
      integer NL 
      do 10 i=1,N
      do 10 j=1,M
   10 AR(i,j) = NL+i+j
                   
      end 


      subroutine ansyes(name)
      character*7 name
      print *,name,'  -  complete'
      end
      subroutine ansno(name)
      character*7 name
      print *,name,'  -  ***error'
      end
