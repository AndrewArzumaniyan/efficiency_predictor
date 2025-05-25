      program TEMPL4

c    TESTING template CLAUSE .       

      print *,'===START OF templ4======================='
C --------------------------------------------------
c 441 TEMPLATE arrA4[BLOCK][BLOCK][BLOCK][BLOCK]
c                     arrB[i][j][k][l] WITH arrA[i+2][ j][k][ l+3] 
c                     ALIGN arrC[i][j] WITH arrA[i+2][2][3][ l+3] 
      call templ441
C --------------------------------------------------
c 442 TEMPLATE arrA1[BLOCK][BLOCK][BLOCK][BLOCK]
c                     ALIGN arrB[i][j][k][l] WITH  arrA[l][i][j][k] 
c                     ALIGN arrC[i][j][k][l] WITH [i+2][ j][k][ l+3] 
      call templ442
C --------------------------------------------------
      print *,'=== END OF templ4 ======================='
      end

C ----------------------------------------------------templ441
c 441 TEMPLATE arrA4[BLOCK][BLOCK][BLOCK][BLOCK]
c                     arrB[i][j][k][l] WITH arrA[i+2][ j][k][ l+3] 
c                     ALIGN arrC[i][j] WITH arrA[i+2][2][3][ l+3] 
      subroutine templ441
      integer, parameter :: AN1=7,AN2=7,AN3=7,AN4=7
      integer, parameter :: BN1=2,BN2=2,BN3=2,BN4=2
      integer, parameter :: CN1=4,CN2=4
      integer, parameter :: NL=10000,ER=100000

c     parameters for ALIGN arrB[i][j][n][m] WITH arrA4(k1i*i+li,k2j*j+lj,k3n*n+ln,k4m*m+lm)                                               
      integer, parameter :: k1i=1,k2i=0,k3i=0,k4i=0,li=2
      integer, parameter :: k1j=0,k2j=1,k3j=0,k4j=0,lj=0
      integer, parameter :: k1n=0,k2n=0,k3n=1,k4n=0,ln=0
      integer, parameter :: k1m=0,k2m=0,k3m=0,k4m=1,lm=3
c     parameters for ALIGN arrC[i][j] WITH arrA4(kc1i*i+lci,lcj,lcn,kc2m*j+lcm)                                               
      integer, parameter :: kc1i=1,kc2i=0,kc3i=0,kc4i=0,lci=2
      integer, parameter :: kc1j=0,kc2j=0,kc3j=0,kc4j=0,lcj=2
      integer, parameter :: kc1n=0,kc2n=0,kc3n=0,kc4n=0,lcn=3
      integer, parameter :: kc1m=0,kc2m=1,kc3m=0,kc4m=0,lcm=3

      character*9 tname
      integer, allocatable :: C2(:,:), B4(:,:,:,:)
      integer s,cs,erri,i,j,n,m,ia,ja,na,ma,ib,jb,nb,mb,ic,jc,nc,mc

cdvm$ template A4(AN1,AN2,AN3,AN4)              
cdvm$ ALIGN B4(i,j,n,m) WITH A4(k1i*i+li,k2j*j+lj,k3n*n+ln,k4m*m+lm)
cdvm$ ALIGN C2(i,j) WITH A4(kc1i*i+lci,lcj,lcn,kc2m*j+lcm)
cdvm$ distribute A4(BLOCK,BLOCK,BLOCK,BLOCK)    

      tname='templ441'
      allocate (C2(CN1,CN2),B4(BN1,BN2,BN3,BN4))
      erri= ER
      NNL=NL 
!dvm$ actual (erri)
!dvm$ region


*dvm$ parallel (m,n,j,i) on B4(i,j,n,m)
      do m=1,BN4
          do n=1,BN3
             do j=1,BN2
                do i=1,BN1
                    B4(i,j,n,m) =(i*NL/10+j*NL/100+n*NL/1000+m)     
                enddo
             enddo
          enddo
      enddo

*dvm$ parallel (j,i) on C2(i,j)
      do j=1,CN2
          do i=1,CN1
            C2(i,j) =(i*NL+j)     
          enddo
      enddo

*dvm$  parallel (m,n,j,i) on A4(i,j,n,m),private(ib,jb,nb,mb,ic,jc,erri)
       do m=1,AN4
          do n=1,AN3
            do j=1,AN2
              do i=1,AN1
                      if ( 
     *                  ((i-li) .eq.(((i-li)/k1i) * k1i)) .and.
     *                  ((j-lj) .eq.(((j-lj)/k2j) *k2j)) .and.
     *                  ((n-ln) .eq.(((n-ln)/k3n) * k3n)) .and.
     *                  ((m-lm) .eq.(((m-lm)/k4m) *k4m)) .and.
     *                  (((i-li)/k1i) .gt. 0)  .and.
     *                  (((j-lj)/k2j) .gt. 0)  .and.
     *                  (((n-ln)/k3n) .gt. 0)  .and.
     *                  (((m-lm)/k4m) .gt. 0)  .and.
     *                  (((i-li)/k1i) .le. BN1)  .and.
     *                  (((j-lj)/k2j) .le. BN2)  .and.
     *                  (((n-ln)/k3n) .le. BN3)  .and.
     *                  (((m-lm)/k4m) .le. BN4)
     *                  )  then 
                        ib = (i-li)/k1i
                        jb = (j-lj)/k2j
                        nb = (n-ln)/k3n
                        mb = (m-lm)/k4m
                        if  (B4(ib,jb,nb,mb).eq.
     *                      (ib*NL/10+jb*NL/100+nb*NL/1000+mb))then     
                        else
                            erri = i*NL/10 + j*NL/100+ n*NL/1000+ m
                        endif
                      endif 
             if (
     *          (j .eq. lcj) .and. (n .eq. lcn) .and.
     *          ((i-lci) .eq.(((i-lci)/kc1i) * kc1i)) .and.
     *          ((m-lcm) .eq.(((m-lcm)/kc2m) *kc2m)) .and.
     *          (((i-lci)/kc1i) .gt. 0)  .and.
     *          (((m-lcm)/kc2m) .gt. 0)  .and.
     *          (((i-lci)/kc1i) .le. CN1)  .and.
     *          (((m-lcm)/kc2m) .le. CN2))  then 
                ic = (i-lci)/kc1i
                jc = (m-lcm)/kc2m  
                if (C2(ic,jc) .eq.(ic*NL+jc)) then     
                else
                    erri = i
                endif
             endif 
              enddo
            enddo
          enddo
      enddo

!dvm$ end region
!dvm$ get_actual (erri)

      if (erri .eq.ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      deallocate (C2,B4)

      end
C ----------------------------------------------------templ442
c 442 TEMPLATE arrA1[BLOCK][BLOCK][BLOCK][BLOCK]
c                     ALIGN arrB[i][j][k][l] WITH  arrA[l][i][j][k] 
c                     ALIGN arrC[i][j][k][l] WITH [i+2][ j][k][ l+3] 
    
      subroutine templ442
      integer, parameter :: AN1=7,AN2=7,AN3=7,AN4=7
      integer, parameter :: BN1=2,BN2=2,BN3=2,BN4=2
      integer, parameter :: CN1=4,CN2=4,CN3=4,CN4=4
      integer, parameter :: NL=10000,ER=100000
c     parameters for ALIGN arrB[i][j][n][m] WITH arrA4(k4i*m+li,k1j*i+lj,k2n*j+ln,k3m*n+lm)                                              
      integer, parameter :: k1i=0,k2i=0,k3i=0,k4i=1,li=0
      integer, parameter :: k1j=1,k2j=0,k3j=0,k4j=0,lj=0
      integer, parameter :: k1n=0,k2n=1,k3n=0,k4n=0,ln=0
      integer, parameter :: k1m=0,k2m=0,k3m=1,k4m=0,lm=0
c     parameters for ALIGN arrC[i][j][n][m] WITH arrA4(kc1i*i+lci,kc2j*j+lcj,kc3n*n+lcn,kc4m*m+lcm)                                               
      integer, parameter :: kc1i=1,kc2i=0,kc3i=0,kc4i=0,lci=2
      integer, parameter :: kc1j=0,kc2j=1,kc3j=0,kc4j=0,lcj=0
      integer, parameter :: kc1n=0,kc2n=0,kc3n=1,kc4n=0,lcn=0
      integer, parameter :: kc1m=0,kc2m=0,kc3m=0,kc4m=1,lcm=3

      character*9 tname
      integer, allocatable :: C4(:,:,:,:),B4(:,:,:,:)
      integer s,cs,erri,i,j,n,m,ia,ja,na,ma,ib,jb,nb,mb,ic,jc,nc,mc
cdvm$ template A4(AN1,AN2,AN3,AN4)              
cdvm$ ALIGN B4(i,j,n,m) WITH A4(k4i*m+li,k1j*i+lj,k2n*j+ln,k3m*n+lm)
cdvm$ ALIGN C4(i,j,n,m) WITH A4(kc1i*i+lci,kc2j*j+lcj,
cdvm$*kc3n*n+lcn,kc4m*m+lcm)
cdvm$ distribute A4(BLOCK,BLOCK,BLOCK,BLOCK)    

      tname='templ442'
      allocate (C4(CN1,CN2,CN3,CN4),B4(BN1,BN2,BN3,BN4))
      erri= ER
      NNL=NL 

!dvm$ actual (erri)
!dvm$ region

*dvm$ parallel (m,n,j,i) on B4(i,j,n,m)
      do m=1,BN4
          do n=1,BN3
             do j=1,BN2
                do i=1,BN1
                    B4(i,j,n,m) =(i*NL/10+j*NL/100+n*NL/1000+m)     
                enddo
             enddo
          enddo
      enddo

*dvm$ parallel (m,n,j,i) on C4(i,j,n,m)
      do m=1,CN4
          do n=1,CN3
             do j=1,CN2
                do i=1,CN1
                    C4(i,j,n,m) =(i*NL/10+j*NL/100+n*NL/1000+m)     
                enddo
             enddo
          enddo
      enddo

*dvm$ parallel (m,n,j,i) on A4(i,j,n,m),
*dvm$*private(ib,jb,nb,mb,ic,jc,nc,mc,erri)
      do m=1,AN4
          do n=1,AN3
            do j=1,AN2
              do i=1,AN1
                      if ( 
     *                  ((i-li) .eq.(((i-li)/k4i) * k4i)) .and.
     *                  ((j-lj) .eq.(((j-lj)/k1j) *k1j)) .and.
     *                  ((n-ln) .eq.(((n-ln)/k2n) * k2n)) .and.
     *                  ((m-lm) .eq.(((m-lm)/k3m) *k3m)) .and.
     *                  (((i-li)/k4i) .gt. 0)  .and.
     *                  (((j-lj)/k1j) .gt. 0)  .and.
     *                  (((n-ln)/k2n) .gt. 0)  .and.
     *                  (((m-lm)/k3m) .gt. 0)  .and.
     *                  (((i-li)/k4i) .le. BN4)  .and.
     *                  (((j-lj)/k1j) .le. BN1)  .and.
     *                  (((n-ln)/k2n) .le. BN2)  .and.
     *                  (((m-lm)/k3m) .le. BN3)
     *                  )  then 
                        mb = (i-li)/k4i
                        ib = (j-lj)/k1j
                        jb = (n-ln)/k2n
                        nb = (m-lm)/k3m
                        if  (B4(ib,jb,nb,mb).eq.
     *                      (ib*NL/10+jb*NL/100+nb*NL/1000+mb))then     
                        else
                            erri = i*NL/10 + j*NL/100+ n*NL/1000+ m
                        endif
                      endif 
                      if ( 
     *                  ((i-lci) .eq.(((i-lci)/kc1i) * kc1i)) .and.
     *                  ((j-lcj) .eq.(((j-lcj)/kc2j) *kc2j)) .and.
     *                  ((n-lcn) .eq.(((n-lcn)/kc3n) * kc3n)) .and.
     *                  ((m-lcm) .eq.(((m-lcm)/kc4m) *kc4m)) .and.
     *                  (((i-lci)/kc1i) .gt. 0)  .and.
     *                  (((j-lcj)/kc2j) .gt. 0)  .and.
     *                  (((n-lcn)/kc3n) .gt. 0)  .and.
     *                  (((m-lcm)/kc4m) .gt. 0)  .and.
     *                  (((i-lci)/kc1i) .le. BN1)  .and.
     *                  (((j-lcj)/kc2j) .le. BN2)  .and.
     *                  (((n-lcn)/kc3n) .le. BN3)  .and.
     *                  (((m-lcm)/kc4m) .le. BN4)
     *                  )  then 
                        ic = (i-lci)/kc1i
                        jc = (j-lcj)/kc2j
                        nc = (n-lcn)/kc3n
                        mc = (m-lcm)/kc4m
                        if (C4(ic,jc,nc,mc) .eq.     
     *                      (ic*NL/10+jc*NL/100+nc*NL/1000+mc))then     
                        else
                            erri = i*NL/10 + j*NL/100+ n*NL/1000+ m
                        endif
             endif 
              enddo
            enddo
          enddo
      enddo

!dvm$ end region
!dvm$ get_actual(erri)

      if (erri .eq.ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      deallocate (C4,B4)

      end
C -------------------------------------------------

      subroutine ansyes(name)
      character*9 name
      print *,name,'  -  complete'
      end
      subroutine ansno(name)
      character*9 name
      print *,name,'  -  ***error'
      end
         
