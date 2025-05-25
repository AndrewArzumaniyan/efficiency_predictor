       program TEMPL11

c    TESTING template CLAUSE .       

      print *,'===START OF templ11======================'
C --------------------------------------------------
c 111 TEMPLATE arrA1[BLOCK]   ALIGN arrB[i] WITH arrA[i+4] 
c                             ALIGN arrC[i] WITH arrA[2*i+4] 
      call templ111
C --------------------------------------------------
c 121 TEMPLATE arrA1[BLOCK]   ALIGN arrB[][i]  WITH arrA[i]	
c                             ALIGN arrC[i][ ] WITH arrA[2*i+1]
      call templ121
C --------------------------------------------------
      print *,'=== END OF templ11 ======================'
      end

C ----------------------------------------------------templ111
c 111   TEMPLATE arrA[BLOCK]  ALIGN arrB[i] WITH arrA[i+4]  
c                             ALIGN arrC[i] WITH arrA[2*i+4] 
      subroutine templ111
      integer, parameter :: AN1=14,CN1=4,BN1=8,PN = 4,NL=1000,ER=10000
c     parameters for ALIGN arrB[i] WITH arrA[k1i * i + li]                                                 
      integer, parameter :: k1i=1,k2i=0,li=4
c     parameters for ALIGN arrC[i] WITH arrA[kc1i * i + lci]                                                 
      integer, parameter :: kc1i=2,kc2i=0,lci=4
      character*9 tname
      integer, allocatable :: C1(:),B1(:)
      integer erri,i,ib,ic

cdvm$ template A1(AN1)               
cdvm$ ALIGN B1(i) WITH A1(k1i * i + li)
cdvm$ ALIGN C1(i) WITH A1(kc1i * i + lci)
cdvm$ distribute A1(BLOCK)    

      tname='templ111'
      allocate (C1(CN1),B1(BN1))
      erri= ER
      NNL=NL 
!dvm$ actual (erri)
!dvm$ region
*dvm$ parallel (i) on B1(i)
      do i=1,BN1
            B1(i) =i     
      enddo

*dvm$ parallel (i) on C1(i)
      do i=1,CN1
            C1(i) =i     
      enddo

*dvm$ parallel (i) on A1(i), private (ib,erri,ic)
      do i=1,AN1
             if (((i-li) .eq.(((i-li)/k1i) * k1i)) .and.
     *          (((i-li)/k1i) .gt. 0)  .and.
     *          (((i-li)/k1i) .le. BN1)) then
                ib = (i-li)/k1i
                if (B1(ib) .eq.(ib)) then     
                else
                    erri = i
                endif 
             endif 
             if (((i-lci) .eq.(((i-lci)/kc1i) * kc1i)) .and.
     *          (((i-lci)/kc1i) .gt. 0)  .and.
     *          (((i-lci)/kc1i) .le. CN1)) then
                ic = (i-lci)/kc1i
                if (C1(ic) .eq.(ic)) then     
                else
                    erri = i
                endif 
             endif 
      enddo

!dvm$ end region
!dvm$ get_actual (erri)

      if (erri .eq.ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      deallocate (C1,B1)

      end
C ----------------------------------------------------templ121
c   121	TEMPLATE arrA1[BLOCK]	
c       	ALIGN arrB[][i]  WITH arrA[i]	
c           ALIGN arrC[i][ ] WITH arrA[2*i+1]
      subroutine templ121
      integer, parameter :: AN1=9,CN1=4,CN2=4,BN1=8,BN2=8
      integer, parameter :: NL=1000,ER=10000
c     parameters for ALIGN arrB(*,i) WITH arrA[k1i*i+li]                                                
      integer, parameter :: k1i=1,k2i=0,li=0
c     parameters for ALIGN arrC(i,*) WITH arrA[kc1i*i+lci]                                                
      integer, parameter :: kc1i=2,kc2i=0,lci=1
      character*9 tname
      integer, allocatable :: C2(:,:),B2(:,:)
      integer erri,i,ib,jb,ic,jc

cdvm$ template A1(AN1)               
cdvm$ ALIGN B2(*,i) WITH A1(k1i*i+li)
cdvm$ ALIGN C2(i,*) WITH A1(kc1i*i+lci)
cdvm$ distribute A1(BLOCK)    

      tname='templ121'
      allocate (C2(CN1,CN2),B2(BN1,BN2))
      erri= ER
      NNL=NL 

!dvm$ actual (erri)
!dvm$ region

*dvm$ parallel (i,j) on B2(i,j)
      do i=1,BN1
          do j=1,BN2
            B2(i,j) =(i*NL+j)     
          enddo
      enddo

*dvm$ parallel (i,j) on C2(i,j)
      do i=1,CN1
          do j=1,CN2
            C2(i,j) =(i*NL+j)     
          enddo
      enddo

*dvm$ parallel (i) on A1(i), private (j,ib,jb,erri,jc,ic,k)
      do i=1,AN1
          do j=1,BN1
             if (  
     *          ((i-li) .eq.(((i-li)/k1i) * k1i)) .and.
     *          (((i-li)/k1i) .gt. 0)  .and.
     *          (((i-li)/k1i) .le. BN2)  )then
                ib = j
                jb = (i-li)/k1i
                if (B2(ib,jb) .eq.(ib*NL+jb)) then     
                else
                    erri = i*NL/10+j
                endif 
             endif   
          enddo
          do k=1,CN2
             if (  
     *          ((i-lci) .eq.(((i-lci)/kc1i) * kc1i)) .and.
     *          (((i-lci)/kc1i) .gt. 0)  .and.
     *          (((i-lci)/kc1i) .le. CN1)  )then
                jc = k
                ic = (i-lci)/kc1i
                if (C2(ic,jc) .eq.(ic*NL+jc)) then     
                else
                    erri = i*NL/10+j
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
      deallocate (C2,B2)

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
         
