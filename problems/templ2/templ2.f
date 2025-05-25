      program TEMPL2

c    TESTING template CLAUSE .       

      print *,'===START OF templ2======================='
C --------------------------------------------------
c 211 TEMPLATE arrA2[BLOCK][BLOCK]
c                     ALIGN arrB[i] WITH arrA[1][i] 
c                     ALIGN arrC[i][j] WITH arrA[2*i+2][j] 
      call templ211
C --------------------------------------------------
c 221 TEMPLATE arrA1[BLOCK][BLOCK]
c                     ALIGN arrB[i][j] WITH arrA[i+4][j+4] 
c                     ALIGN arrC[i][j] WITH arrA[i+1][j+1] 
      call templ221
C --------------------------------------------------
      print *,'=== END OF templ2 ======================='
      end

C ----------------------------------------------------templ211
c 211 TEMPLATE arrA2[BLOCK][BLOCK]
c                     ALIGN arrB[i] WITH arrA[1][i] 
c                     ALIGN arrC[i][j] WITH arrA[2*i+2][j] 
      subroutine templ211
      integer, parameter :: AN1=14,AN2=14,CN1=4,CN2=4,BN1=8
      integer, parameter :: NL=1000,ER=10000
c     parameters for ALIGN arrB[i] WITH arrA(1,i)                                                
      integer, parameter :: k1i=0,k2i=0,li=1,k1j=1,k2j=0,lj=0
c     parameters for ALIGN arrC[i][j] WITH arrA[kc1i * i + lci][kc2j * j + lcj]
      integer, parameter :: kc1i=2,kc2i=0,lci=2,kc1j=0,kc2j=1,lcj=0
      character*9 tname
      integer, allocatable :: C2(:,:),B1(:)
      integer erri,i,ib,ic,jc

cdvm$ template A2(AN1,AN2)               
cdvm$ ALIGN B1(i) WITH A2(1,i)
cdvm$ ALIGN C2(i,j) WITH A2(kc1i * i + lci,kc2j * j + lcj)
cdvm$ distribute A2(BLOCK,BLOCK)    

      tname='templ211'
      allocate (C2(CN1,CN2),B1(BN1))
      erri= ER
      NNL=NL 

!dvm$ actual(erri)
!dvm$ region

*dvm$ parallel (i) on B1(i)
      do i=1,BN1
            B1(i) =i     
      enddo

*dvm$ parallel (j,i) on C2(i,j)
      do j=1,CN2
          do i=1,CN1
            C2(i,j) =(i*NL+j)     
          enddo
      enddo

*dvm$ parallel (j,i) on A2(i,j), private (ib,erri,ic,jc)
      do j=1,AN2
          do i=1,AN1
             if ((i .eq. 1) ) then
               if( 
     *           (j .le. BN1)  
     *           )  then 
                        ib = j
                if (B1(ib) .eq.(ib)) then     
                else
                    erri = i
                endif 
               endif 
             endif   
             if (((i-lci) .eq.(((i-lci)/kc1i) * kc1i)) .and.
     *          ((j-lcj) .eq.(((j-lcj)/kc2j) *kc2j)) .and.
     *          (((i-lci)/kc1i) .gt. 0)  .and.
     *          (((j-lcj)/kc2j) .gt. 0)  .and.
     *          (((i-lci)/kc1i) .le. CN1)  .and.
     *          (((j-lcj)/kc2j) .le. CN2))  then 
                ic = (i-lci)/kc1i
                jc = (j-lcj)/kc2j  
                if (C2(ic,jc) .eq.(ic*NL+jc)) then     
                else
                    erri = i
                endif
             endif 
          enddo
      enddo

!dvm$ end region
!dvm$ get_actual (erri)

      if (erri .eq.ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      deallocate (C2,B1)

      end
C ----------------------------------------------------templ221
c 221 TEMPLATE arrA1[BLOCK][BLOCK]
c                     ALIGN arrB[i][j] WITH arrA[i+4][j+4] 
c                     ALIGN arrC[i][j] WITH arrA[i+1][j+1] 
      subroutine templ221
      integer, parameter :: AN1=14,AN2=14,CN1=4,CN2=4,BN1=8,BN2=8
      integer, parameter :: NL=1000,ER=10000
c     parameters for ALIGN arrB[i][j] WITH arrA[k1i * i + li][k2j * j + lj]                                                 
      integer, parameter :: k1i=1,k2i=0,li=4,k1j=0,k2j=1,lj=4
c     parameters for ALIGN arrC[i][j] WITH arrA[kc1i * i + lci][kc2j * j + lcj]                                                 
      integer, parameter :: kc1i=1,kc2i=0,lci=1,kc1j=0,kc2j=1,lcj=1
      character*9 tname
      integer, allocatable :: C2(:,:),B2(:,:)
      integer erri,i,ib,jb,ic,jc

cdvm$ template A2(AN1,AN2)               
cdvm$ ALIGN B2(i,j) WITH A2(k1i * i + li,k2j * j + lj)
cdvm$ ALIGN C2(i,j) WITH A2(kc1i * i + lci,kc2j * j + lcj)
cdvm$ distribute A2(BLOCK,BLOCK)    

      tname='templ221'
      allocate (C2(CN1,CN2),B2(BN1,BN2))
      erri= ER
      NNL=NL 
!dvm$ actual (erri)
!dvm$ region


*dvm$ parallel (j,i) on B2(i,j)
      do j=1,BN2
          do i=1,BN1
            B2(i,j) =(i*NL+j)     
          enddo
      enddo

*dvm$ parallel (j,i) on C2(i,j)
      do j=1,CN2
          do i=1,CN1
            C2(i,j) =(i*NL+j)     
          enddo
      enddo

*dvm$ parallel (j,i) on A2(i,j),private (ib,ic,erri,jb,jc)
      do j=1,AN2
          do i=1,AN1
             if (((i-li) .eq.(((i-li)/k1i) * k1i)) .and.
     *          ((j-lj) .eq.(((j-lj)/k2j) *k2j)) .and.
     *          (((i-li)/k1i) .gt. 0)  .and.
     *          (((j-lj)/k2j) .gt. 0)  .and.
     *          (((i-li)/k1i) .le. BN1)  .and.
     *          (((j-lj)/k2j) .le. BN2))  then 
                ib = (i-li)/k1i
                jb = (j-lj)/k2j  
                if (B2(ib,jb) .eq.(ib*NL+jb)) then     
                else
                    erri = i
                endif 
             endif 
             if (((i-lci) .eq.(((i-lci)/kc1i) * kc1i)) .and.
     *          ((j-lcj) .eq.(((j-lcj)/kc2j) *kc2j)) .and.
     *          (((i-lci)/kc1i) .gt. 0)  .and.
     *          (((j-lcj)/kc2j) .gt. 0)  .and.
     *          (((i-lci)/kc1i) .le. CN1)  .and.
     *          (((j-lcj)/kc2j) .le. CN2))  then 
                ic = (i-lci)/kc1i
                jc = (j-lcj)/kc2j  
                if (C2(ic,jc) .eq.(ic*NL+jc)) then     
                else
                    erri = i
                endif
             endif 
          enddo
      enddo
!dvm$ end region
!dvm$ get_actual(erri)

      if (erri .eq.ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 

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
   
