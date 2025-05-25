      program DISTRMIX3

!    Testing DISTRIBUTE and REDISTRIBUTE directives
!            GEN_BLOCK, WGT_BLOCK, MULT_BLOCK, BLOCK, *  distributions

      integer PROCESSORS_RANK, PROCESSORS_SIZE 
      integer psize(3), rank

      PROCESSORS_RANK() = 3
      PROCESSORS_SIZE(i) = 1

      print *,'===START OF distrmix3========================'

C -------------------------------------------------
c 31  DISTRIBUTE arrA3[MULT_BLOCK][MULT_BLOCK][MULT_BLOCK]
c              REDISTRIBUTE [WGT_BLOCK][WGT_BLOCK][WGT_BLOCK] 
c              REDISTRIBUTE [MULT_BLOCK][MULT_BLOCK][MULT_BLOCK]
      call distrmix31
C -------------------------------------------------
c 32 DISTRIBUTE arrA3[WGT_BLOCK][WGT_BLOCK][WGT_BLOCK] 
c              REDISTRIBUTE [MULT_BLOCK][MULT_BLOCK][MULT_BLOCK]
c              REDISTRIBUTE [WGT_BLOCK][WGT_BLOCK][WGT_BLOCK] 
      call distrmix32
C -------------------------------------------------
c 33  DISTRIBUTE arrA3[MULT_BLOCK][WGT_BLOCK][MULT_BLOCK]
c              REDISTRIBUTE [WGT_BLOCK][MULT_BLOCK][WGT_BLOCK] 
c              REDISTRIBUTE [MULT_BLOCK][MULT_BLOCK][MULT_BLOCK]
      call distrmix33
C -------------------------------------------------
c 34  DISTRIBUTE arrA3[WGT_BLOCK][WGT_BLOCK][MULT_BLOCK]
c              REDISTRIBUTE [WGT_BLOCK][MULT_BLOCK][WGT_BLOCK] 
c              REDISTRIBUTE [MULT_BLOCK][WGT_BLOCK][WGT_BLOCK]
      call distrmix34
C -------------------------------------------------
c 35  DISTRIBUTE arrA3[MULT_BLOCK][WGT_BLOCK][BLOCK]
c              REDISTRIBUTE [WGT_BLOCK][BLOCK][MULT_BLOCK] 
c              REDISTRIBUTE [BLOCK][MULT_BLOCK][WGT_BLOCK]
      call distrmix35
C -------------------------------------------------
c 36  DISTRIBUTE arrA3[WGT_BLOCK][MULT_BLOCK][BLOCK]
c              REDISTRIBUTE [MULT_BLOCK][BLOCK][WGT_BLOCK] 
c              REDISTRIBUTE [BLOCK][WGT_BLOCK][MULT_BLOCK]
      call distrmix36
C -------------------------------------------------
c 37  DISTRIBUTE arrA3[MULT_BLOCK][BLOCK][BLOCK]
c              REDISTRIBUTE [BLOCK][BLOCK][WGT_BLOCK] 
c              REDISTRIBUTE [BLOCK][WGT_BLOCK][MULT_BLOCK]
      call distrmix37
C -------------------------------------------------
c 38  DISTRIBUTE arrA3[MULT_BLOCK][*][WGT_BLOCK]
c              REDISTRIBUTE [*][*][*] 
c              REDISTRIBUTE [WGT_BLOCK][MULT_BLOCK][*]
      call distrmix38
C -------------------------------------------------
c 39  DISTRIBUTE arrA3[MULT_BLOCK][BLOCK][*]
c              REDISTRIBUTE [WGT_BLOCK][*][WGT_BLOCK] 
c              REDISTRIBUTE [*][MULT_BLOCK][*]
      call distrmix39
C -------------------------------------------------
c 310  DISTRIBUTE arrA3[WGT_BLOCK][*][*]
c              REDISTRIBUTE [MULT_BLOCK][*][WGT_BLOCK] 
c              REDISTRIBUTE [*][WGT_BLOCK][MULT_BLOCK]
      call distrmix310                         
C -------------------------------------------------

      rank = PROCESSORS_RANK()

      do i=1,rank
         psize(i)=PROCESSORS_SIZE(i)
         if (psize(i) >  4) then ! may be temporary
             goto 1 
         endif
      enddo

C -------------------------------------------------
       if
     > ((psize(1) == 1 .and. psize(2) == 1 .and. psize(3) == 1)  !range 1 1 1
     >.or.
     >  (psize(1) == 1 .and. psize(2) == 2 .and. psize(3) == 3)  !range 1 2 3
     >.or.
     >  (psize(1) == 2 .and. psize(2) == 3 .and. psize(3) == 2)  !range 2 3 2
     >.or.
     >  (psize(1) == 3 .and. psize(2) == 4 .and. psize(3) == 1)  !range 3 4 1
     >.or.
     >  (psize(1) == 4 .and. psize(2) == 1 .and. psize(3) == 4)) !range 4 1 4
     >then
c 311  DISTRIBUTE arrA3[MULT_BLOCK][MULT_BLOCK][MULT_BLOCK]
c             REDISTRIBUTE [GEN_BLOCK][GEN_BLOCK][GEN_BLOCK] 
c             REDISTRIBUTE [MULT_BLOCK][MULT_BLOCK][MULT_BLOCK]
        call distrmix311 (psize)
      endif
C-------------------------------------------------
      if
     > ((psize(1) == 1 .and. psize(2) == 1 .and. psize(3) == 1)  !range 1 1 1
     >.or.
     >  (psize(1) == 1 .and. psize(2) == 4 .and. psize(3) == 4)  !range 1 4 4
     >.or.
     >  (psize(1) == 2 .and. psize(2) == 4 .and. psize(3) == 2)  !range 2 4 2
     >.or.
     >  (psize(1) == 3 .and. psize(2) == 1 .and. psize(3) == 3)  !range 3 1 3
     >.or.
     >  (psize(1) == 4 .and. psize(2) == 2 .and. psize(3) == 2)) !range 4 2 2
     >then
c 312 DISTRIBUTE arrA3[GEN_BLOCK][GEN_BLOCK][GEN_BLOCK]
c             REDISTRIBUTE [MULT_BLOCK][WGT_BLOCK][MULT_BLOCK]
c             REDISTRIBUTE [GEN_BLOCK][GEN_BLOCK][BLOCK]
        call distrmix312 (psize)
      endif
C-------------------------------------------------
c 313  DISTRIBUTE arrA3[BLOCK][BLOCK][MULT_BLOCK]
c              REDISTRIBUTE [BLOCK][GEN_BLOCK][BLOCK] 
c              REDISTRIBUTE [MULT_BLOCK][MULT_BLOCK][WGT_BLOCK]
         call distrmix313 (psize)
C -------------------------------------------------
c 314 DISTRIBUTE arrA3[WGT_BLOCK][BLOCK][MULT_BLOCK]
c              REDISTRIBUTE [GEN_BLOCK][BLOCK][GEN_BLOCK] 
c              REDISTRIBUTE [MULT_BLOCK][BLOCK][WGT_BLOCK]
         call distrmix314 (psize)
C -------------------------------------------------
c 315  DISTRIBUTE arrA3[MULT_BLOCK][WGT_BLOCK][WGT_BLOCK]
c              REDISTRIBUTE [BLOCK][GEN_BLOCK][GEN_BLOCK] 
c              REDISTRIBUTE [MULT_BLOCK][WGT_BLOCK][MULT_BLOCK]
         call distrmix315 (psize)
C -------------------------------------------------
c 316  DISTRIBUTE arrA3[WGT_BLOCK][WGT_BLOCK][MULT_BLOCK]
c              REDISTRIBUTE [GEN_BLOCK][GEN_BLOCK][BLOCK] 
c              REDISTRIBUTE [MULT_BLOCK][WGT_BLOCK][MULT_BLOCK]
          call distrmix316 (psize)
C -------------------------------------------------
c 317  DISTRIBUTE arrA3[GEN_BLOCK][*][GEN_BLOCK]
c              REDISTRIBUTE [MULT_BLOCK][WGT_BLOCK][*]
c              REDISTRIBUTE [*][GEN_BLOCK][BLOCK] 
         call distrmix317 (psize)
C -------------------------------------------------
      if
     > ((psize(1) == 1 .and. psize(2) == 1 .and. psize(3) == 1)  !range 1 1 1
     >.or.
     >  (psize(1) == 1 .and. psize(2) == 4 .and. psize(3) == 3)  !range 1 4 3
     >.or.
     >  (psize(1) == 2 .and. psize(2) == 3 .and. psize(3) == 2)  !range 2 3 2
     >.or.
     >  (psize(1) == 3 .and. psize(2) == 1 .and. psize(3) == 4)  !range 3 1 4
     >.or.
     >  (psize(1) == 4 .and. psize(2) == 2 .and. psize(3) == 2)) !range 4 2 2
     >then
c 318  DISTRIBUTE arrA3[GEN_BLOCK][GEN_BLOCK][GEN_BLOCK]  
c                REDISTRIBUTE [WGT_BLOCK][WGT_BLOCK][WGT_BLOCK]
c                REDISTRIBUTE [GEN_BLOCK][GEN_BLOCK][GEN_BLOCK]  
        call distrmix318 (psize)
      endif
C -------------------------------------------------
      if
     > ((psize(1) == 1 .and. psize(2) == 2 .and. psize(3) == 1)  !range 1 2 1
     >.or.
     >  (psize(1) == 2 .and. psize(2) == 2 .and. psize(3) == 2)  !range 2 2 2
     >.or.
     >  (psize(1) == 3 .and. psize(2) == 2 .and. psize(3) == 2)  !range 3 2 2
     >.or.
     >  (psize(1) == 4 .and. psize(2) == 4 .and. psize(3) == 1)) !range 4 4 1
     >then

c 319 DISTRIBUTE arrA3[WGT_BLOCK][WGT_BLOCK][WGT_BLOCK] 
c               REDISTRIBUTE [GEN_BLOCK][GEN_BLOCK][GEN_BLOCK] 
c               REDISTRIBUTE arrA3[WGT_BLOCK][WGT_BLOCK][WGT_BLOCK]
        call distrmix319 (psize)
      endif
C -------------------------------------------------
c 320  DISTRIBUTE arrA3[BLOCK][GEN_BLOCK][GEN_BLOCK]  
c                REDISTRIBUTE arrB3[WGT_BLOCK][BLOCK][WGT_BLOCK] 
        call distrmix320 (psize)
C -------------------------------------------------
c 321 DISTRIBUTE arrA3[WGT_BLOCK][BLOCK][WGT_BLOCK]  ! static
c                REDISTRIBUTE [GEN_BLOCK][BLOCK][GEN_BLOCK]
        call distrmix321 (psize)
C -------------------------------------------------
c 322  DISTRIBUTE arrA3[GEN_BLOCK][BLOCK][GEN_BLOCK]
c                REDISTRIBUTE [BLOCK][WGT_BLOCK][BLOCK]
         call distrmix322 (psize)
C -------------------------------------------------
c 323 DISTRIBUTE arrA3[BLOCK][WGT_BLOCK][BLOCK]
c                REDISTRIBUTE [GEN_BLOCK][GEN_BLOCK][*]
         call distrmix323 (psize)
C -------------------------------------------------
c 324 DISTRIBUTE arrA3[GEN_BLOCK][GEN_BLOCK][*]    
c                REDISTRIBUTE [*][WGT_BLOCK][WGT_BLOCK]
c                REDISTRIBUTE [GEN_BLOCK][*][GEN_BLOCK]
         call distrmix324 (psize)
C -------------------------------------------------
c 325  DISTRIBUTE arrA3 [*][GEN_BLOCK][GEN_BLOCK]  
c                REDISTRIBUTE [*][WGT_BLOCK][*]
c                REDISTRIBUTE [GEN_BLOCK][GEN_BLOCK][GEN_BLOCK]
        call distrmix325 (psize)
C -------------------------------------------------
C
 1     print *,'=== END OF distrmix3 ========================= '    

      end

C ----------------------------------------------------distrmix31
c 31  DISTRIBUTE arrA3[MULT_BLOCK][MULT_BLOCK][MULT_BLOCK]
c              REDISTRIBUTE [WGT_BLOCK][WGT_BLOCK][WGT_BLOCK] 

c              REDISTRIBUTE [MULT_BLOCK][MULT_BLOCK][MULT_BLOCK]

      subroutine distrmix31
      
      integer, parameter :: AN1=32,AN2=32,AN3=32,NL=1000,ER=10000
      integer :: erri=ER,i,j,k

      integer, parameter :: m11 = 4, m21 = 8, m31 = 2 
      integer, parameter :: m12 = 2, m22 = 4, m32 = 4 

      double precision, dimension(7) :: 
     >            WB1=(/2.0,1.5,4.,3.0, 2., 3., 2./)     
      double precision, dimension(8)::
     >            WB2=(/1.0,1.,2.,2.0,1.,1.,2.,2./)     
      double precision, dimension(7) :: 
     >            WB3=(/2.0,2.,2.6,3.0, 1., 1.5, 1./)     

      integer, allocatable :: A3(:,:,:)
      character(12), parameter :: tname='distrmix31  '

               
!dvm$ distribute A3(MULT_BLOCK(m11),MULT_BLOCK(m21),MULT_BLOCK(m31))   
!dvm$ dynamic A3

      allocate (A3(AN1,AN2,AN3))

!dvm$ region
!dvm$ parallel (i,j,k) on A3(i,j,k)
      do i=1,AN1
          do j=1,AN2
             do k=1,AN3
                A3(i,j,k) = i*NL/10 + j*NL/100 + k    
             enddo
          enddo
      enddo
!dvm$ end region

!dvm$ redistribute 
!dvm$* A3(WGT_BLOCK(WB1,7),WGT_BLOCK(WB2,8),WGT_BLOCK(WB3,7))   

!dvm$ region
!dvm$ parallel (i,j,k) on A3(i,j,k)
      do i=1,AN1
          do j=1,AN2
             do k=1,AN3
                A3(i,j,k) = A3(i,j,k) + 2    
             enddo
          enddo
      enddo
!dvm$ end region

!dvm$ redistribute A3(MULT_BLOCK(m12),MULT_BLOCK(m22),MULT_BLOCK(m32))   

!dvm$ actual (erri)

!dvm$ region
!dvm$ parallel (i,j,k) on A3(i,j,k), reduction( min(erri) )
      do i=1,AN1
          do j=1,AN2
            do k=1,AN3
                if (A3(i,j,k) /= (i*NL/10 + j*NL/100 + k) + 2) then     
                    erri = min(erri,i*NL/10 + j*NL/100 + k)
                endif 
            enddo
          enddo
      enddo
!dvm$ end region
     
!dvm$ get_actual (erri)

      if (erri  == ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 

      deallocate (A3)

      end subroutine distrmix31

C ----------------------------------------------------distrmix32
c 32 DISTRIBUTE arrA3[WGT_BLOCK][WGT_BLOCK][WGT_BLOCK] 
c              REDISTRIBUTE [MULT_BLOCK][MULT_BLOCK][MULT_BLOCK]
c              REDISTRIBUTE [WGT_BLOCK][WGT_BLOCK][WGT_BLOCK] 

      subroutine distrmix32

      integer, parameter :: AN1=16,AN2=16,AN3=12,NL=1000,ER=10000
      integer :: erri=ER,i,j,k

      integer, parameter :: m1 = 2, m2 = 4, m3 = 4

      double precision, dimension(6) :: 
     >            WB1=(/2.0,5.,0.,3.0, 2., 3./)     
      double precision, dimension(8)::
     >            WB2=(/1.2,2.,4.,2.5,3.,1.,3.,2./)     
      double precision, dimension(7) :: 
     >            WB3=(/2.3,1.2,4.6,3.0, 1.5, 2.5, 1.2/)     

      integer, allocatable :: A3(:,:,:)
      character(12), parameter :: tname='distrmix32  '

!dvm$ distribute 
!dvm$* A3(WGT_BLOCK(WB1,6),WGT_BLOCK(WB2,8),WGT_BLOCK(WB3,7))   
!dvm$ dynamic A3

      allocate (A3(AN1,AN2,AN3))
     
      A3 = 10 

!dvm$ actual (A3)

!dvm$ region
!dvm$ parallel (i,j,k) on A3(i,j,k)
      do i=1,AN1
          do j=1,AN2
             do k=1,AN3
                A3(i,j,k) = A3(i,j,k) + i*NL/10 + j*NL/100 + k    
             enddo
          enddo
      enddo
!dvm$ end region

!dvm$ redistribute A3(MULT_BLOCK(m1),MULT_BLOCK(m2),MULT_BLOCK(m3))   

!dvm$ region
!dvm$ parallel (i,j,k) on A3(i,j,k)
      do i=1,AN1
          do j=1,AN2
             do k=1,AN3
                A3(i,j,k) = A3(i,j,k) + 2    
             enddo
          enddo
      enddo
!dvm$ end region

!dvm$ redistribute 
!dvm$* A3(WGT_BLOCK(WB2,8),WGT_BLOCK(WB3,7),WGT_BLOCK(WB1,6))   

!dvm$ actual (erri)

!dvm$ region
!dvm$ parallel (i,j,k) on A3(i,j,k), reduction( min( erri ) )
      do i=1,AN1
          do j=1,AN2
            do k=1,AN3
                if (A3(i,j,k) /= (i*NL/10 + j*NL/100 + k + 12)) then     
                    erri = min(erri,i*NL/10 + j*NL/100 + k)
                endif 
            enddo
          enddo
      enddo
!dvm$ end region
     
!dvm$ get_actual (erri)

      if (erri == ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 

      deallocate (A3)

      end subroutine distrmix32

C----------------------------------------------------distrmix33
c 33  DISTRIBUTE arrA3[MULT_BLOCK][WGT_BLOCK][MULT_BLOCK]
c              REDISTRIBUTE [WGT_BLOCK][MULT_BLOCK][WGT_BLOCK] 
c              REDISTRIBUTE [MULT_BLOCK][MULT_BLOCK][MULT_BLOCK]

      subroutine distrmix33

      integer, parameter :: AN1=12,AN2=18,AN3=20,NL=1000,ER=10000
      integer :: erri=ER,i,j,k

      integer, parameter :: m11 = 2, m21 = 3, m31 = 2
      integer, parameter :: m12 = 6, m22 = 9, m32 = 5
 
      double precision, dimension(7) :: 
     >            WB1=(/2.2, 2.4, 4., 2.5, 3.5, 1.,3./)     
      double precision, dimension(6)::
     >            WB2=(/1.2, 2., 2.5, 3., 1.5, 3./)     
      double precision, dimension(5) :: 
     >            WB3=(/4.3, 2.2, 2.6, 2.0, 2.5/)     
                                        
      integer, allocatable :: A3(:,:,:)
      character(12), parameter :: tname='distrmix33  '
 
!dvm$ distribute A3(MULT_BLOCK(m11),WGT_BLOCK(WB2,6),MULT_BLOCK(m31))   
!dvm$ dynamic A3

      allocate (A3(AN1,AN2,AN3))

      A3 = 7

!dvm$ actual (A3)

!dvm$ region
!dvm$ parallel (i,j,k) on A3(i,j,k)
      do i=1,AN1
          do j=1,AN2
             do k=1,AN3
                A3(i,j,k) = A3(i,j,k) + i*NL/10 + j*NL/100 + k    
             enddo
         enddo
      enddo
!dvm$ end region

!dvm$ redistribute 
!dvm$*    A3(WGT_BLOCK(WB1,7),MULT_BLOCK(m21),WGT_BLOCK(WB3,5))   

!dvm$ region
!dvm$ parallel (i,j,k) on A3(i,j,k)
      do i=1,AN1
         do j=1, AN2
            do k=1,AN3
               A3(i,j,k) = A3(i,j,k) - 5
             enddo
          enddo
      enddo
!dvm$ end region
    
!dvm$ redistribute
!dvm$*    A3(MULT_BLOCK(m12),MULT_BLOCK(m22),MULT_BLOCK(m32))   

!dvm$ actual (erri)

!dvm$ region
!dvm$ parallel (i,j,k) on A3(i,j,k), reduction( min(erri) )
      do i=1,AN1
           do j=1,AN2
             do k=1,AN3
                 if (A3(i,j,k) /= (i*NL/10 + j*NL/100 + k + 2)) then     
                     erri = min(erri,i*NL/10 + j*NL/100 + k)
                 endif 
             enddo
          enddo
      enddo
!dvm$ end region
    
!dvm$ get_actual (erri)

      if (erri == ER) then     
         call ansyes(tname)
      else
         call ansno(tname)
      endif 

      deallocate (A3)

      end subroutine distrmix33

C------------------------------------------------------distrmix34
c 34  DISTRIBUTE arrA3[WGT_BLOCK][WGT_BLOCK][MULT_BLOCK]
c              REDISTRIBUTE [WGT_BLOCK][MULT_BLOCK][WGT_BLOCK] 
c              REDISTRIBUTE [MULT_BLOCK][WGT_BLOCK][WGT_BLOCK]

      subroutine distrmix34

      integer, parameter :: AN1=35,AN2=28,AN3=16,NL=1000,ER=10000
      integer :: erri=ER,i,j,k

      integer, parameter :: m1 = 7, m2 = 7, m3 = 4

      double precision, dimension(8) :: 
     >            WB1=(/2., 2., 4., 2.7, 3.5, 2., 1., 3./)     
      double precision, dimension(6)::
     >            WB2=(/12., 2.5, 3., 1.5, 3., 2./)     
      double precision, dimension(7) :: 
     >            WB3=(/4.,3., 2.2, 2.6, 2.0, 2.5, 1./)     
                                        
      integer, allocatable :: A3(:,:,:)
      character(12), parameter :: tname='distrmix34  '

!dvm$ distribute :: A3
!dvm$ dynamic A3

      allocate (A3(AN1,AN2,AN3))

!dvm$ redistribute
!dvm$*  A3(WGT_BLOCK(WB1,8),WGT_BLOCK(WB2,6),MULT_BLOCK(m3))   

        A3 = 0

!dvm$ actual (A3)

!dvm$ region
!dvm$ parallel (i,j,k) on A3(i,j,k)
      do i=1,AN1
          do j=1,AN2
             do k=1,AN3
               A3(i,j,k) = A3(i,j,k) + i*NL/10 + j*NL/100 + k    
             enddo                               
         enddo
      enddo
!dvm$ end region

!dvm$ redistribute
!dvm$*  A3(WGT_BLOCK(WB2,6),MULT_BLOCK(m2),WGT_BLOCK(WB3,7))   

!dvm$ region
!dvm$ parallel (i,j,k) on A3(i,j,k)
       do i=1,AN1
          do j=1,AN2
            do k=1,AN3
                A3(i,j,k) = A3(i,j,k)*2     
            enddo
          enddo
       enddo
!dvm$ end region
 
!dvm$ redistribute
!dvm$*  A3(MULT_BLOCK(m1),WGT_BLOCK(WB3,7),WGT_BLOCK(WB1,8))   

!dvm$ actual (erri)

!dvm$ region
!dvm$ parallel (i,j,k) on A3(i,j,k), reduction( min(erri) )
      do i=1,AN1
          do j=1,AN2
            do k=1,AN3
                if (A3(i,j,k) /= (i*NL/10 + j*NL/100 + k) * 2) then     
                    erri = min(erri,i*NL/10 + j*NL/100 + k)
               endif 
            enddo
          enddo
      enddo
!dvm$ end region

!dvm$ get_actual (erri)

      if (erri == ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 

      deallocate (A3)

      end subroutine distrmix34

C------------------------------------------------------distrmix35
c 35   DISTRIBUTE arrA3[MULT_BLOCK][WGT_BLOCK][BLOCK]
c               REDISTRIBUTE [WGT_BLOCK][BLOCK][MULT_BLOCK] 
c               REDISTRIBUTE [BLOCK][MULT_BLOCK][WGT_BLOCK]

      subroutine distrmix35

      integer, parameter :: AN1= 10, AN2=21, AN3=32,NL=1000,ER=10000
      integer :: erri=ER,i,j,k

      integer, parameter :: m1 = 2, m2 = 3, m3 = 4 
 
      double precision, dimension(7) :: 
     >            WB1=(/2., 4., 3., 2.5, 5., 1., 2./)     
      double precision, dimension(10)::
     >            WB2=(/1., 2., 5., 3., 1., 3., 2., 3., 2., 1./)     
      double precision, dimension(8) :: 
     >            WB3=(/2.3, 2.2, 1.6, 1., 2.0, 2.5, 3., 2./)     
                                        
      integer A3(AN1,AN2,AN3)    ! static array
      character(12), parameter :: tname='distrmix35  '

!dvm$ distribute A3(MULT_BLOCK(m1),WGT_BLOCK(WB2,10),BLOCK)   
!dvm$ dynamic A3

!dvm$ region
!dvm$ parallel (i,j,k) on A3(i,j,k)
      do i=1,AN1
        do j=1,AN2
           do k=1,AN3
               A3(i,j,k) = i*NL/10 + j*NL/100 + k + 5   
            enddo
         enddo
      enddo
!dvm$ end region

!dvm$ redistribute
!dvm$*    A3(WGT_BLOCK(WB1,7),BLOCK,MULT_BLOCK(m3))   

!dvm$ region
!dvm$ parallel (i,j,k) on A3(i,j,k)
      do i=1,AN1
        do j=1,AN2
           do k=1,AN3
               A3(i,j,k) = A3(i,j,k) + 2
            enddo
          enddo
      enddo
!dvm$ end region

!dvm$ redistribute
!dvm$*    A3(BLOCK, MULT_BLOCK(m2),WGT_BLOCK(WB3,8))   

!dvm$ actual (erri)

!dvm$ region
!dvm$ parallel (i,j,k) on A3(i,j,k), reduction( min(erri) )
      do i=1,AN1
         do j=1,AN2
            do k=1,AN3
                if (A3(i,j,k) /= (i*NL/10 + j*NL/100 + k + 7)) then     
                   erri = min(erri,i*NL/10 + j*NL/100 + k)
                endif 
            enddo
          enddo
      enddo
!dvm$ end region
     
!dvm$ get_actual (erri)

      if (erri == ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 

      end subroutine distrmix35

C------------------------------------------------------distrmix36
c 36  DISTRIBUTE arrA3[WGT_BLOCK][MULT_BLOCK][BLOCK]
c             REDISTRIBUTE [MULT_BLOCK][BLOCK][WGT_BLOCK] 
c             REDISTRIBUTE [BLOCK][WGT_BLOCK][MULT_BLOCK]
 
      subroutine distrmix36

      integer, parameter :: AN1=16,AN2=28,AN3=16,NL=1000,ER=10000
      integer :: erri=ER,i,j,k

      integer, parameter :: m1 = 2, m2 = 7, m3 = 4

      double precision, dimension(8) :: 
     >           WB1=(/1.2,2.,4.,2.5,3.,1.,3.,2./)     
      double precision, dimension(7)::
     >           WB2=(/2.,2.,4.,2.5,3.,1.,3./)     
      double precision, dimension(7) :: 
     >           WB3=(/2.5,2.2,4.2,2.0, 1.5, 3.5, 1.2/)     

      integer, allocatable :: A3(:,:,:)
      character(12), parameter :: tname='distrmix36  '

!dvm$ distribute A3(WGT_BLOCK(WB1,8),MULT_BLOCK(m2),BLOCK)   
!dvm$ dynamic A3

      allocate (A3(AN1,AN2,AN3))

!dvm$ region
!dvm$ parallel (i,j,k) on A3(i,j,k)
      do i=1,AN1
        do j=1,AN2
           do k=1,AN3
               A3(i,j,k) = (i*NL/10 + j*NL/100 + k) * 3     
           enddo  
         enddo
      enddo
!dvm$ end region

!dvm$ redistribute
!dvm$*    A3(MULT_BLOCK(m1),BLOCK,WGT_BLOCK(WB3,7))   

!dvm$ region
!dvm$ parallel (i,j,k) on A3(i,j,k)
      do i=1,AN1
        do j=1,AN2
           do k=1,AN3
               A3(i,j,k) = A3(i,j,k) * 2
            enddo
          enddo
      enddo
!dvm$ end region

!dvm$ redistribute
!dvm$*    A3(BLOCK, WGT_BLOCK(WB2,7),MULT_BLOCK(m3))   

!dvm$ actual (erri)

!dvm$ region
!dvm$ parallel (i,j,k) on A3(i,j,k), reduction( min(erri) )
      do i=1,AN1
         do j=1,AN2
            do k=1,AN3
                if (A3(i,j,k) /= (i*NL/10 + j*NL/100 + k) * 6) then     
                   erri = min(erri,i*NL/10 + j*NL/100 + k)
                endif 
            enddo
          enddo
      enddo
!dvm$ end region

!dvm$ get_actual (erri)

      if (erri == ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 

      deallocate (A3)

      end subroutine distrmix36
 
C -----------------------------------------------------distrmix37
c 37  DISTRIBUTE arrA3[MULT_BLOCK][BLOCK][BLOCK]
c              REDISTRIBUTE [BLOCK][BLOCK][WGT_BLOCK] 
c              REDISTRIBUTE [BLOCK][WGT_BLOCK][MULT_BLOCK]

      subroutine distrmix37

      integer, parameter :: AN1=10,AN2=10,AN3=30,NL=1000,ER=10000
      integer :: erri=ER,i,j,k

      integer, parameter :: m1 = 2, m2 = 5, m3 = 3

      double precision, dimension(6) :: 
     >           WB2=(/4., 2.5, 3., 1., 3., 2./)     
      double precision, dimension(8)::
     >           WB3=(/1.,2.,3.,3.5, 4., 1., 3., 2./)     

      integer, allocatable :: A3(:,:,:)
      character(12), parameter :: tname='distrmix37  '

!dvm$ distribute A3(MULT_BLOCK(m1),BLOCK,BLOCK)   
!dvm$ dynamic A3

      allocate (A3(AN1,AN2,AN3))

!dvm$ region
!dvm$ parallel (i,j,k) on A3(i,j,k)
      do i=1,AN1
        do j=1,AN2
           do k=1,AN3
               A3(i,j,k) = (i*NL/10 + j*NL/100 + k) * 3     
           enddo  
         enddo
      enddo
!dvm$ end region

!dvm$ redistribute
!dvm$*    A3(BLOCK,BLOCK,WGT_BLOCK(WB3,8))   

!dvm$ region
!dvm$ parallel (i,j,k) on A3(i,j,k)
      do i=1,AN1
        do j=1,AN2
           do k=1,AN3
               A3(i,j,k) = A3(i,j,k) * 2
            enddo
          enddo
      enddo
!dvm$ end region

!dvm$ redistribute
!dvm$*    A3(BLOCK, WGT_BLOCK(WB2,6),MULT_BLOCK(m3))   

!dvm$ actual (erri)

!dvm$ region
!dvm$ parallel (i,j,k) on A3(i,j,k), reduction( min(erri) )
      do i=1,AN1
         do j=1,AN2
            do k=1,AN3
                if (A3(i,j,k) /= (i*NL/10 + j*NL/100 + k) * 6) then     
                   erri = min(erri,i*NL/10 + j*NL/100 + k)
                endif 
            enddo
          enddo
      enddo
!dvm$ end region

!dvm$ get_actual (erri)

      if (erri == ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 

      deallocate (A3)

      end subroutine distrmix37
 
C------------------------------------------------------distrmix38
c 38  DISTRIBUTE arrA3[MULT_BLOCK][*][WGT_BLOCK]
c              REDISTRIBUTE [*][*][*] 
c              REDISTRIBUTE [WGT_BLOCK][MULT_BLOCK][*]

      subroutine distrmix38

      integer, parameter :: AN1=16,AN2=16,AN3=16,NL=1000,ER=10000
      integer :: erri=ER,i,j,k
    
      integer, parameter :: m1 = 2, m2 = 1, m3 = 4

      double precision, dimension(11) :: 
     >   WB=(/2.2, 3.,3., 2.5, 2., 1., 4., 2., 1., 5., 2./)     

      integer, allocatable :: A3(:,:,:)
      character(12), parameter :: tname='distrmix38  '

!dvm$ distribute A3(MULT_BLOCK(m1),*,WGT_BLOCK(WB,11))   
!dvm$ dynamic A3

      allocate (A3(AN1,AN2,AN3))

      A3 = 5

!dvm$ actual (A3)

!dvm$ region
!dvm$ parallel (i,j,k) on A3(i,j,k)
      do i=1,AN1
        do j=1,AN2
           do k=1,AN3
               A3(i,j,k) = A3(i,j,k) + (i*NL/10 + j*NL/100 + k) 
           enddo  
         enddo
      enddo
!dvm$ end region

!dvm$ redistribute  A3(*,*,*)   

!dvm$ region
!dvm$ parallel (i,j,k) on A3(i,j,k)
      do i=1,AN1
        do j=1,AN2
           do k=1,AN3
               A3(i,j,k) = A3(i,j,k) + 2
            enddo
          enddo
      enddo
!dvm$ end region

!dvm$ redistribute
!dvm$*    A3(WGT_BLOCK(WB,8),MULT_BLOCK(m2),*)   

!dvm$ actual (erri)

!dvm$ region
!dvm$ parallel (i,j,k) on A3(i,j,k), reduction( min(erri) )
      do i=1,AN1
         do j=1,AN2
            do k=1,AN3
              if (A3(i,j,k) /= (i*NL/10 + j*NL/100 + k) + 7) then     
                   erri = min(erri,i*NL/10 + j*NL/100 + k)
               endif 
            enddo
          enddo
      enddo
!dvm$ end region

!dvm$ get_actual (erri)

      if (erri == ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 

      deallocate (A3)

      end subroutine distrmix38

C ----------------------------------------------------distrmix39
C 39  DISTRIBUTE arrA3[MULT_BLOCK][BLOCK][*]
c             REDISTRIBUTE [WGT_BLOCK][*][WGT_BLOCK] 
c             REDISTRIBUTE [*][MULT_BLOCK][*]

      subroutine distrmix39

      integer, parameter :: AN1=18,AN2=6,AN3=30,NL=1000,ER=10000
      integer :: erri=ER,i,j,k
   
      integer, parameter :: m1 = 3, m2 = 2, m3 = 5
      
      double precision, dimension(11) :: 
     >   WB=(/3.2, 2., 2., 1.5, 4., 2., 3., 2.5, 1.6, 3., 2./)     

      integer, allocatable :: A3(:,:,:)
      character(12), parameter :: tname='distrmix39  '

!dvm$ distribute A3(MULT_BLOCK(m1),BLOCK,*)   
!dvm$ dynamic A3

      allocate (A3(AN1,AN2,AN3))

      A3 = 7

!dvm$ actual (A3)

!dvm$ region
!dvm$ parallel (i,j,k) on A3(i,j,k)
      do i=1,AN1
        do j=1,AN2
          do k=1,AN3
             A3(i,j,k) = A3(i,j,k) + (i*NL/10 + j*NL/100 + k) 
          enddo
        enddo
      enddo
!dvm$ end region

!dvm$ redistribute
!dvm$*    A3(WGT_BLOCK(WB,11),*,WGT_BLOCK(WB,7))   

!dvm$ actual (erri)

!dvm$ region
!dvm$ parallel (i,j,k) on A3(i,j,k)
      do i=1,AN1
          do j=1,AN2
             do k=1,AN3
                A3(i,j,k) = A3(i,j,k) + 4    
             enddo
          enddo
      enddo
!dvm$ end region
 
!dvm$ redistribute
!dvm$*    A3(*,MULT_BLOCK(m2),*)   

!dvm$ region
!dvm$ parallel (i,j,k) on A3(i,j,k), reduction( min(erri) )
      do i=1,AN1
          do j=1,AN2
            do k=1,AN3
                if (A3(i,j,k) /= (i*NL/10 + j*NL/100 + k) + 11) then     
                    erri = min(erri, i*NL/10 + j*NL/100 + k)
                endif 
            enddo
          enddo
      enddo
!dvm$ end region

!dvm$ get_actual (erri)

      if (erri == ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif

      deallocate (A3)

      end subroutine distrmix39

C ----------------------------------------------------distrmix310
c 310  DISTRIBUTE arrA3[WGT_BLOCK][*][*]
c             REDISTRIBUTE [MULT_BLOCK][*][WGT_BLOCK] 
c             REDISTRIBUTE [*][WGT_BLOCK][MULT_BLOCK]

      subroutine distrmix310

      integer, parameter :: AN1=25,AN2=35,AN3=10,NL=1000,ER=10000
      integer :: erri=ER,i,j,k

      integer, parameter :: m1 = 5, m2 = 7, m3 = 2

      double precision, dimension(12) :: 
     >  WB=(/3., 1., 2., 1.5, 3., 4., 3., 2.5, 1.6, 3., 1.2, 1./)     

      integer, allocatable :: A3(:,:,:)
      character(12), parameter :: tname='distrmix310 '

!dvm$ distribute A3(WGT_BLOCK(WB,12),*,*)   
!dvm$ dynamic A3

      allocate (A3(AN1,AN2,AN3))

!dvm$ region
!dvm$ parallel (i,j,k) on A3(i,j,k)
      do i=1,AN1
          do j=1,AN2
             do k=1,AN3
                A3(i,j,k) = i*NL/10 + j*NL/100 + k    
             enddo
          enddo
      enddo
!dvm$ end region

!dvm$ redistribute
!dvm$*    A3(MULT_BLOCK(m1), *, WGT_BLOCK(WB,8))   

!dvm$ region
!dvm$ parallel (i,j,k) on A3(i,j,k)
      do i=1,AN1
          do j=1,AN2
             do k=1,AN3
                A3(i,j,k) = A3(i,j,k) - 2    
            enddo
         enddo
      enddo
!dvm$ end region

!dvm$ redistribute
!dvm$*    A3(*,MULT_BLOCK(m2),WGT_BLOCK(WB,8))   

!dvm$ actual (erri)

!dvm$ region
!dvm$ parallel (i,j,k) on A3(i,j,k), reduction( min(erri) )
      do i=1,AN1
         do j=1,AN2
             do k=1,AN3
                if (A3(i,j,k) /= (i*NL/10 + j*NL/100 + k) - 2) then     
                   erri = min(erri,i*NL/10 + j*NL/100 + k)
                endif 
             enddo
          enddo
      enddo
!dvm$ end region

!dvm$ get_actual (erri)

      if (erri == ER) then     
          call ansyes(tname)
      else
         call ansno(tname)
      endif

      deallocate (A3)

      end subroutine distrmix310

C ----------------------------------------------------distrmix311
c 311  DISTRIBUTE arrA3[MULT_BLOCK][MULT_BLOCK][MULT_BLOCK]       range 1 1 1
c              REDISTRIBUTE [GEN_BLOCK][GEN_BLOCK][GEN_BLOCK]     range 1 2 3
c              REDISTRIBUTE [MULT_BLOCK][MULT_BLOCK][MULT_BLOCK]  range 2 3 2
c                                                                 range 3 2 2
c                                                                 range 4 1 4
      subroutine distrmix311 (psize)
      integer psize(3)

      integer, parameter :: AN1=15,AN2=15,AN3=28,NL=1000,ER=10000
      integer :: erri=ER,i,j,k

      integer, parameter :: m11 = 3, m21 = 5, m31 = 4 
      integer, parameter :: m12 = 5, m22 = 3, m32 = 7 

      integer, dimension(1) :: BSi111=(/15/)      !range 1 1 1  
      integer, dimension(1) :: BSj111=(/15/)     
      integer, dimension(1) :: BSk111=(/28/)     

      integer, dimension(1) :: BSi1=(/15/)        !range 1 2 3  
      integer, dimension(2) :: BSj1=(/8,7/)     
      integer, dimension(3) :: BSk1=(/12,10,6/)     

      integer, dimension(2) :: BSi2=(/4,11/)      !range 2 3 2  
      integer, dimension(3) :: BSj2=(/7,5,3/)     
      integer, dimension(2) :: BSk2=(/10,18/)     

      integer, dimension(3) :: BSi3=(/2,8,5/)     !range 3 4 1  
      integer, dimension(4) :: BSj3=(/3,2,6,4/)     
      integer, dimension(1) :: BSk3=(/28/)     

      integer, dimension(4) :: BSi4=(/1,2,4,8/)   !range 4 1 4  
      integer, dimension(1) :: BSj4=(/15/)     
      integer, dimension(4) :: BSk4=(/12,4,6,6/)     

      integer, allocatable :: A3(:,:,:)
      character(12), parameter :: tname='distrmix311 '

!dvm$ distribute A3(MULT_BLOCK(m11),MULT_BLOCK(m21),MULT_BLOCK(m31))   
!dvm$ dynamic A3

      allocate (A3(AN1,AN2,AN3))

!dvm$ region
!dvm$ parallel (i,j,k) on A3(i,j,k)
      do i=1,AN1
          do j=1,AN2
             do k=1,AN3
                A3(i,j,k) = i*NL/10 + j*NL/100 + k    
             enddo
          enddo
      enddo
!dvm$ end region

!!!!dvm$ redistribute A3(GEN_BLOCK(BSi),BLOCK,GEN_BLOCK(BSk))   

      select case(psize(1))
      case(1) 
        if (psize(2) == 1 .and. psize(3) == 1) then
!dvm$ redistribute
!dvm$*    A3(GEN_BLOCK(BSi111),GEN_BLOCK(BSj111),GEN_BLOCK(BSk111))   
        else
!dvm$ redistribute
!dvm$*    A3(GEN_BLOCK(BSi1),GEN_BLOCK(BSj1),GEN_BLOCK(BSk1))   
        endif
      case(2)
!dvm$ redistribute
!dvm$*    A3(GEN_BLOCK(BSi2),GEN_BLOCK(BSj2),GEN_BLOCK(BSk2))   
      case(3)
!dvm$ redistribute
!dvm$*    A3(GEN_BLOCK(BSi3),GEN_BLOCK(BSj3),GEN_BLOCK(BSk3))   
      case(4)
!dvm$ redistribute
!dvm$*    A3(GEN_BLOCK(BSi4),GEN_BLOCK(BSj4),GEN_BLOCK(BSk4))   
      case default 
         goto 10
      endselect  

!dvm$ region
!dvm$ parallel (i,j,k) on A3(i,j,k)
      do i=1,AN1
          do j=1,AN2
             do k=1,AN3
                A3(i,j,k) = A3(i,j,k) * 2    
             enddo
          enddo
      enddo
!dvm$ end region

!dvm$ redistribute A3(MULT_BLOCK(m12),MULT_BLOCK(m22),MULT_BLOCK(m32))   

!dvm$ actual (erri)

!dvm$ region
!dvm$ parallel (i,j,k) on A3(i,j,k), reduction( min(erri) )
      do i=1,AN1
          do j=1,AN2
            do k=1,AN3
                if (A3(i,j,k) /= (i*NL/10 + j*NL/100 + k) * 2) then     
                    erri = min(erri,i*NL/10 + j*NL/100 + k)
                endif 
            enddo
          enddo
      enddo
!dvm$ end region
     
!dvm$ get_actual (erri)

      if (erri == ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 

 10   deallocate (A3)

      end subroutine distrmix311

C ----------------------------------------------------distrmix312
c 312  DISTRIBUTE arrA3[GEN_BLOCK][GEN_BLOCK][GEN_BLOCK]          range 1 1 1
c              REDISTRIBUTE [MULT_BLOCK][WGT_BLOCK][MULT_BLOCK]   range 1 4 4
c              REDISTRIBUTE [GEN_BLOCK][GEN_BLOCK][BLOCK]         range 2 4 2
c                                                                 range 3 1 3
c                                                                 range 4 2 2
      subroutine distrmix312 (psize)
      integer psize(3)

      integer, parameter :: AN1=24,AN2=10,AN3=24,NL=1000,ER=10000
      integer :: erri=ER,i,j,k

      integer, parameter :: m1 = 3, m2 = 2, m3 = 4 

      double precision, dimension(10) :: 
     >  WB=(/2., 2.5, 3., 4., 3.5, 2.5, 2.6, 3., 2.2, 3./)     

      integer, dimension(1) :: BSi111=(/24/)        !range 1 1 1  
      integer, dimension(1) :: BSj111=(/10/)     
      integer, dimension(1) :: BSk111=(/24/)     
                                                              
      integer, dimension(1) :: BSi11=(/24/)         !range 1 4 4  
      integer, dimension(1) :: BSi12=(/24/)           
      integer, dimension(4) :: BSj11=(/3,2,4,1/)     
      integer, dimension(4) :: BSj12=(/4,2,1,3/)     
      integer, dimension(4) :: BSk11=(/10,4,3,7/)     
      integer, dimension(4) :: BSk12=(/5,6,7,6/)     

      integer, dimension(2) :: BSi21=(/14,10/)      !range 2 4 2  
      integer, dimension(2) :: BSi22=(/8,16/)     
      integer, dimension(4) :: BSj21=(/3,2,1,4/)     
      integer, dimension(4) :: BSj22=(/5,3,2,0/)     
      integer, dimension(2) :: BSk21=(/20,4/)     
      integer, dimension(2) :: BSk22=(/16,8/)          

      integer, dimension(3) :: BSi31=(/8,12,4/)     !range 3 1 3  
      integer, dimension(3) :: BSi32=(/3,10,11/)     
      integer, dimension(1) :: BSj31=(/10/)     
      integer, dimension(1) :: BSj32=(/10/)     
      integer, dimension(3) :: BSk31=(/7,9,8/)     
      integer, dimension(3) :: BSk32=(/4,6,14/)     

      integer, dimension(4) :: BSi41=(/2,6,12,4/)   !range 4 2 2  
      integer, dimension(4) :: BSi42=(/3,2,9,10/)     
      integer, dimension(2) :: BSj41=(/6,4/)     
      integer, dimension(2) :: BSj42=(/10,0/)     
      integer, dimension(2) :: BSk41=(/14,10/)     
      integer, dimension(2) :: BSk42=(/6,18/)     

      integer, allocatable :: A3(:,:,:)
      character(12), parameter :: tname='distrmix312 '

!dvm$ distribute :: A3   
!dvm$ dynamic A3

      allocate (A3(AN1,AN2,AN3))

!!!!dvm$ redistribute A3(GEN_BLOCK(BSi1),GEN_BLOCK(BSj1),GEN_BLOCK(BSk1))   
      select case(psize(1))
      case(1) 
        if (psize(2) == 1 .and. psize(3) == 1) then
!dvm$ redistribute
!dvm$*    A3(GEN_BLOCK(BSi111),GEN_BLOCK(BSj111),GEN_BLOCK(BSk111))   
        else
!dvm$ redistribute
!dvm$*    A3(GEN_BLOCK(BSi11),GEN_BLOCK(BSj11),GEN_BLOCK(BSk11))   
        endif
      case(2)
!dvm$ redistribute
!dvm$*    A3(GEN_BLOCK(BSi21),GEN_BLOCK(BSj21),GEN_BLOCK(BSk21))   
      case(3)
!dvm$ redistribute
!dvm$*    A3(GEN_BLOCK(BSi31),GEN_BLOCK(BSj31),GEN_BLOCK(BSk31))   
      case(4)
!dvm$ redistribute
!dvm$*    A3(GEN_BLOCK(BSi41),GEN_BLOCK(BSj41),GEN_BLOCK(BSk41))   
      case default 
         goto 10
      endselect  

!dvm$ region
!dvm$ parallel (i,j,k) on A3(i,j,k)
      do i=1,AN1
          do j=1,AN2
             do k=1,AN3
                A3(i,j,k) = i*NL/10 + j*NL/100 + k  + 30 
             enddo                                     
          enddo
      enddo
!dvm$ end region

!dvm$ redistribute A3(MULT_BLOCK(m1),WGT_BLOCK(WB,10),MULT_BLOCK(m3))   

!dvm$ region
!dvm$ parallel (i,j,k) on A3(i,j,k)
      do i=1,AN1
          do j=1,AN2
             do k=1,AN3
                A3(i,j,k) = A3(i,j,k) + 30 
             enddo                                     
          enddo
      enddo
!dvm$ end region

!!!!dvm$ redistribute A3(GEN_BLOCK(BSi2),GEN_BLOCK(BSj2),BLOCK)

      select case(psize(1))
      case(1)
        if  (psize(2) == 1 .and. psize(3) == 1) then
!dvm$ redistribute
!dvm$*    A3(GEN_BLOCK(BSi111),GEN_BLOCK(BSj111),BLOCK)   
        else
!dvm$ redistribute
!dvm$*    A3(GEN_BLOCK(BSi12),GEN_BLOCK(BSj12),BLOCK)   
        endif
      case(2)
!dvm$ redistribute
!dvm$*    A3(GEN_BLOCK(BSi22),GEN_BLOCK(BSj22),BLOCK)   
      case(3)
!dvm$ redistribute
!dvm$*    A3(GEN_BLOCK(BSi32),GEN_BLOCK(BSj32),BLOCK)   
      case(4)
!dvm$ redistribute
!dvm$*    A3(GEN_BLOCK(BSi42),GEN_BLOCK(BSj42),BLOCK)   
      case default 
         goto 10
      endselect  

!dvm$ actual (erri)

!dvm$ region
!dvm$ parallel (i,j,k) on A3(i,j,k), reduction( min( erri ) )
      do i=1,AN1
          do j=1,AN2
            do k=1,AN3
                A3(i,j,k) = A3(i,j,k) - 30 
                if (A3(i,j,k) /= (i*NL/10 + j*NL/100 + k + 30)) then     
                    erri = min(erri,i*NL/10 + j*NL/100 + k)
                endif 
            enddo
          enddo
      enddo
!dvm$ end region
     
!dvm$ get_actual (erri)

      if (erri == ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 

 10   deallocate (A3)

      end subroutine distrmix312

C------------------------------------------------------distrmix313
c 313  DISTRIBUTE arrA3[BLOCK][BLOCK][MULT_BLOCK]             
c           REDISTRIBUTE [BLOCK][GEN_BLOCK][BLOCK]            
c           REDISTRIBUTE [MULT_BLOCK][MULT_BLOCK][WGT_BLOCK]  

      subroutine distrmix313 (psize)
      integer psize(3)

      integer, parameter :: AN1=12,AN2=24,AN3=36,NL=1000,ER=10000
      integer :: erri=ER,i,j,k

      integer, parameter :: m1 = 2, m2 = 3, m3 = 4 

      double precision, dimension(9) :: 
     >  WB=(/1., 2.5, 3., 4., 2.5, 2.6, 3.5, 4.2, 3./)     

      integer, dimension(1) :: BSj1=(/24/)        
      integer, dimension(2) :: BSj2=(/21,3/)      
      integer, dimension(3) :: BSj3=(/7,9,8/)     
      integer, dimension(4) :: BSj4=(/10,4,6,4/)  

      integer, allocatable :: A3(:,:,:)
      character(12), parameter :: tname='distrmix313 '

!dvm$ distribute A3(BLOCK, BLOCK, MULT_BLOCK(m3))   
!dvm$ dynamic A3

      allocate (A3(AN1,AN2,AN3))

!dvm$ region
!dvm$ parallel (i,j,k) on A3(i,j,k)
      do i=1,AN1
          do j=1,AN2
             do k=1,AN3
                A3(i,j,k) = i*NL/10 + j*NL/100 + k  + 20       
             enddo                                     
          enddo
      enddo
!dvm$ end region

!!!!dvm$ redistribute A3(BLOCK,GEN_BLOCK(BSj),BLOCK)   
      select case(psize(2))
      case(1)
!dvm$ redistribute A3(BLOCK,GEN_BLOCK(BSj1),BLOCK)   
      case(2)
!dvm$ redistribute A3(BLOCK,GEN_BLOCK(BSj2),BLOCK)   
      case(3)
!dvm$ redistribute A3(BLOCK,GEN_BLOCK(BSj3),BLOCK)   
      case(4)
!dvm$ redistribute A3(BLOCK,GEN_BLOCK(BSj4),BLOCK)   
      case default 
        goto 10
      endselect  

!dvm$ region
!dvm$ parallel (i,j,k) on A3(i,j,k)
      do i=1,AN1
         do j=1,AN2
           do k=1,AN3
              A3(i,j,k) = A3(i,j,k) + 5
           enddo                                     
         enddo
      enddo
!dvm$ end region

!dvm$ redistribute A3(MULT_BLOCK(m1),MULT_BLOCK(m2),WGT_BLOCK(WB,9))   

!dvm$ actual (erri)

!dvm$ region
!dvm$ parallel (i,j,k) on A3(i,j,k), reduction( min( erri ) )
      do i=1,AN1
          do j=1,AN2
            do k=1,AN3
                A3(i,j,k) = A3(i,j,k) - 20 
                if (A3(i,j,k) /= (i*NL/10 + j*NL/100 + k + 5)) then     
                    erri = min(erri,i*NL/10 + j*NL/100 + k)
                endif 
            enddo
          enddo
      enddo
!dvm$ end region
    
!dvm$ get_actual (erri)

      if (erri == ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 

 10   deallocate (A3)

      end subroutine distrmix313

C-----------------------------------------------------distrmix314
c 314 DISTRIBUTE arrA3[WGT_BLOCK][BLOCK][MULT_BLOCK]        
c              REDISTRIBUTE [GEN_BLOCK][BLOCK][GEN_BLOCK]   
c              REDISTRIBUTE [MULT_BLOCK][BLOCK][WGT_BLOCK]  
c                                                            
      subroutine distrmix314 (psize)
      integer psize(3)

      integer, parameter :: AN1=24,AN2=15,AN3=12,NL=1000,ER=10000
      integer :: erri=ER,i,j,k

      integer, parameter :: m1 = 4, m2 = 3, m3 = 2 

      double precision, dimension(10) :: 
     >  WB=(/3., 2., 2., 4., 2., 3., 2.5, 2.6, 1.2, 2./)     

      integer, dimension(1) :: BSi1=(/24/)         
      integer, dimension(2) :: BSi2=(/14,10/)        
      integer, dimension(3) :: BSi3=(/12,8,4/)       
      integer, dimension(4) :: BSi4=(/6,6,5,7/)      

      integer, dimension(1) :: BSj1=(/15/)     
      integer, dimension(2) :: BSj2=(/7,8/)     
      integer, dimension(3) :: BSj3=(/3,4,8/)     
      integer, dimension(4) :: BSj4=(/1,6,3,5/)                

      integer, dimension(1) :: BSk1=(/12/)     
      integer, dimension(2) :: BSk2=(/4,8/)     
      integer, dimension(3) :: BSk3=(/6,2,4/)     
      integer, dimension(4) :: BSk4=(/2,3,6,1/)     

      integer A3(AN1,AN2,AN3)      ! static
      character(12), parameter :: tname='distrmix314 '

!dvm$ distribute A3(WGT_BLOCK(WB,10), BLOCK, MULT_BLOCK(m3))   
!dvm$ dynamic A3

!dvm$ region
!dvm$ parallel (i,j,k) on A3(i,j,k)
      do i=1,AN1
          do j=1,AN2
             do k=1,AN3
                A3(i,j,k) = i*NL/10 + j*NL/100 + k        
             enddo                                     
          enddo
      enddo
!dvm$ end region

!!!!dvm$ redistribute A3(GEN_BLOCK(BSi),BLOCK,GEN_BLOCK(BSk))   

      select case(psize(1))
      case(1)
          select case(psize(3))
          case(1) 
!dvm$ redistribute A3(GEN_BLOCK(BSi1),BLOCK,GEN_BLOCK(BSk1))    
          case(2)
!dvm$ redistribute A3(GEN_BLOCK(BSi1),BLOCK,GEN_BLOCK(BSk2))    
          case(3)
!dvm$ redistribute A3(GEN_BLOCK(BSi1),BLOCK,GEN_BLOCK(BSk3))    
          case(4)
!dvm$ redistribute A3(GEN_BLOCK(BSi1),BLOCK,GEN_BLOCK(BSk4))    
          case default 
              goto 10
          endselect  

      case(2)
          select case(psize(3))
          case(1) 
!dvm$ redistribute A3(GEN_BLOCK(BSi2),BLOCK,GEN_BLOCK(BSk1))    
          case(2)
!dvm$ redistribute A3(GEN_BLOCK(BSi2),BLOCK,GEN_BLOCK(BSk2))    
          case(3)
!dvm$ redistribute A3(GEN_BLOCK(BSi2),BLOCK,GEN_BLOCK(BSk3))    
          case(4)
!dvm$ redistribute A3(GEN_BLOCK(BSi2),BLOCK,GEN_BLOCK(BSk4))    
          case default 
             goto 10
          endselect  

      case(3)
          select case(psize(3))
          case(1) 
!dvm$ redistribute A3(GEN_BLOCK(BSi3),BLOCK,GEN_BLOCK(BSk1))    
          case(2)
!dvm$ redistribute A3(GEN_BLOCK(BSi3),BLOCK,GEN_BLOCK(BSk2))    
          case(3)
!dvm$ redistribute A3(GEN_BLOCK(BSi3),BLOCK,GEN_BLOCK(BSk3))    
          case(4)
!dvm$ redistribute A3(GEN_BLOCK(BSi3),BLOCK,GEN_BLOCK(BSk4))    
          case default 
             goto 10
          endselect  

      case(4)
          select case(psize(3))
          case(1) 
!dvm$ redistribute A3(GEN_BLOCK(BSi4),BLOCK,GEN_BLOCK(BSk1))    
          case(2)
!dvm$ redistribute A3(GEN_BLOCK(BSi4),BLOCK,GEN_BLOCK(BSk2))    
          case(3)
!dvm$ redistribute A3(GEN_BLOCK(BSi4),BLOCK,GEN_BLOCK(BSk3))    
          case(4)
!dvm$ redistribute A3(GEN_BLOCK(BSi4),BLOCK,GEN_BLOCK(BSk4))    
          case default 
             goto 10
          endselect  

      case default 
         goto 10
      endselect  

!dvm$ region
!dvm$ parallel (i,j,k) on A3(i,j,k)
       do i=1,AN1
         do j=1,AN2
           do k=1,AN3
              A3(i,j,k) = i*NL/10 + j*NL/100 + k  + 2
           enddo                                     
         enddo
       enddo
!dvm$ end region

!dvm$ redistribute A3(MULT_BLOCK(m1),BLOCK,MULT_BLOCK(m3))   

!dvm$ actual (erri)

!dvm$ region
!dvm$ parallel (i,j,k) on A3(i,j,k), reduction( min( erri ) )
      do i=1,AN1
          do j=1,AN2
            do k=1,AN3
                if (A3(i,j,k) /= (i*NL/10 + j*NL/100 + k + 2)) then     
                    erri = min(erri,i*NL/10 + j*NL/100 + k)
                endif 
            enddo
          enddo
      enddo
!dvm$ end region
    
!dvm$ get_actual (erri)

      if (erri == ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 

 10   continue

      end subroutine distrmix314

C ----------------------------------------------------distrmix315
c 315  DISTRIBUTE arrA3[MULT_BLOCK][WGT_BLOCK][WGT_BLOCK]        
c              REDISTRIBUTE [BLOCK][GEN_BLOCK][GEN_BLOCK]        
c              REDISTRIBUTE [MULT_BLOCK][WGT_BLOCK][MULT_BLOCK]  
                                                                
      subroutine distrmix315 (psize)
      integer psize(3)

      integer, parameter :: AN1=21, AN2=14, AN3=16, NL=1000,ER=10000           
      integer :: erri=ER,i,j,k

      integer, parameter :: m1 = 3, m2 = 2, m3 = 4 

      double precision, dimension(8) :: 
     >  WB1=(/2., 4., 3., 1., 2.5, 2.6, 2.2, 2./)     
      double precision, dimension(10) :: 
     >  WB2=(/4., 2., 2.5, 4., 2., 3., 3.5, 1.6, 3.2, 2./)     

      integer, dimension(1) :: BSi1=(/21/)         
      integer, dimension(2) :: BSi2=(/14,7/)        
      integer, dimension(3) :: BSi3=(/10,8,3/)       
      integer, dimension(4) :: BSi4=(/6,6,5,4/)      

      integer, dimension(1) :: BSj1=(/14/)     
      integer, dimension(2) :: BSj2=(/3,11/)     
      integer, dimension(3) :: BSj3=(/4,3,7/)     
      integer, dimension(4) :: BSj4=(/2,6,2,4/)                

      integer, dimension(1) :: BSk1=(/16/)     
      integer, dimension(2) :: BSk2=(/4,12/)     
      integer, dimension(3) :: BSk3=(/6,3,7/)     !rem
      integer, dimension(4) :: BSk4=(/2,3,6,5/)     

      integer, allocatable :: A3(:,:,:)
      character(12), parameter :: tname='distrmix315 '

!dvm$ distribute 
!dvm$* A3 (MULT_BLOCK(m1), WGT_BLOCK(WB1,8), WGT_BLOCK(WB2,10))   
!dvm$ dynamic A3

      allocate (A3(AN1,AN2,AN3))

!dvm$ region
!dvm$ parallel (i,j,k) on A3(i,j,k)
      do i=1,AN1
          do j=1,AN2
             do k=1,AN3
                A3(i,j,k) = i*NL/10 + j*NL/100 + k  + 12
             enddo                                     
         enddo
      enddo
!dvm$ end region

!!!!dvm$ redistribute A3(BLOCK,GEN_BLOCK(BSj),GEN_BLOCK(BSk))   

      select case(psize(2))
      case(1)
          select case(psize(3))
          case(1) 
!dvm$ redistribute A3(BLOCK,GEN_BLOCK(BSj1),GEN_BLOCK(BSk1))    
          case(2)
!dvm$ redistribute A3(BLOCK,GEN_BLOCK(BSj1),GEN_BLOCK(BSk2))    
          case(3)
!dvm$ redistribute A3(BLOCK,GEN_BLOCK(BSj1),GEN_BLOCK(BSk3))    
          case(4)
!dvm$ redistribute A3(BLOCK,GEN_BLOCK(BSj1),GEN_BLOCK(BSk4))    
          case default 
             goto 10
          endselect  

      case(2)
          select case(psize(3))
          case(1) 
!dvm$ redistribute A3(BLOCK,GEN_BLOCK(BSj2),GEN_BLOCK(BSk1))    
          case(2)
!dvm$ redistribute A3(BLOCK,GEN_BLOCK(BSj2),GEN_BLOCK(BSk2))    
          case(3)
!dvm$ redistribute A3(BLOCK,GEN_BLOCK(BSj2),GEN_BLOCK(BSk3))    
          case(4)
!dvm$ redistribute A3(BLOCK,GEN_BLOCK(BSj2),GEN_BLOCK(BSk4))    
          case default 
             goto 10
          endselect  

      case(3)
          select case(psize(3))
          case(1) 
!dvm$ redistribute A3(BLOCK,GEN_BLOCK(BSj3),GEN_BLOCK(BSk1))    
          case(2)
!dvm$ redistribute A3(BLOCK,GEN_BLOCK(BSj3),GEN_BLOCK(BSk2))    
          case(3)
!dvm$ redistribute A3(BLOCK,GEN_BLOCK(BSj3),GEN_BLOCK(BSk3))    
          case(4)
!dvm$ redistribute A3(BLOCK,GEN_BLOCK(BSj3),GEN_BLOCK(BSk4))    
          case default 
             goto 10
          endselect  

      case(4)
          select case(psize(3))
          case(1) 
!dvm$ redistribute A3(BLOCK,GEN_BLOCK(BSj4),GEN_BLOCK(BSk1))    
          case(2)
!dvm$ redistribute A3(BLOCK,GEN_BLOCK(BSj4),GEN_BLOCK(BSk2))    
          case(3)
!dvm$ redistribute A3(BLOCK,GEN_BLOCK(BSj4),GEN_BLOCK(BSk3))    
          case(4)
!dvm$ redistribute A3(BLOCK,GEN_BLOCK(BSj4),GEN_BLOCK(BSk4))    
          case default 
             goto 10
          endselect  

      case default 
         goto 10
      endselect  

!dvm$ region
!dvm$ parallel (i,j,k) on A3(i,j,k)
      do i=1,AN1
        do j=1,AN2
          do k=1,AN3
             A3(i,j,k) = A3(i,j,k) + 12
          enddo                                     
        enddo
      enddo
!dvm$ end region

!dvm$ redistribute A3(MULT_BLOCK(m1),WGT_BLOCK(WB1,6),MULT_BLOCK(m3))   

!dvm$ actual (erri)

!dvm$ region
!dvm$ parallel (i,j,k) on A3(i,j,k), reduction( min( erri ) )
      do i=1,AN1
        do j=1,AN2
           do k=1,AN3
               A3(i,j,k) = A3(i,j,k) - 20 
                if (A3(i,j,k) /= (i*NL/10 + j*NL/100 + k + 4)) then     
                   erri = min(erri,i*NL/10 + j*NL/100 + k)
                endif 
            enddo
         enddo
      enddo
!dvm$ end region
  
!dvm$ get_actual (erri)

      if (erri == ER) then     
         call ansyes(tname)
      else
         call ansno (tname)
      endif 

 10   deallocate (A3)

      end subroutine distrmix315

C-----------------------------------------------------distrmix316
c 316  DISTRIBUTE arrA3[WGT_BLOCK][WGT_BLOCK][MULT_BLOCK]         
c              REDISTRIBUTE [GEN_BLOCK][GEN_BLOCK][BLOCK]        
c              REDISTRIBUTE [MULT_BLOCK][WGT_BLOCK][MULT_BLOCK]  
                                                                
      subroutine distrmix316 (psize)
      integer psize(3)

      integer, parameter :: AN1=33,AN2=44,AN3=55,NL=1000,ER=10000
      integer :: erri=ER,i,j,k

      integer, parameter :: m1 = 3, m2 = 11, m3 = 5 

      double precision, dimension (7) ::
     >  WB1=(/3., 2.5, 2., 4., 2.5, 2.0, 3.5/)     
      double precision, dimension(8) :: 
     >  WB2=(/4., 3., 2.5, 2., 2., 3., 3.5, 2.6/)     

      integer, dimension(1) :: BSi1=(/33/)            
      integer, dimension(2) :: BSi2=(/23,10/)         
      integer, dimension(3) :: BSi3=(/12,15,6/)      
      integer, dimension(4) :: BSi4=(/6,13,11,3/)    !rem 

      integer, dimension(1) :: BSj1=(/44/)     
      integer, dimension(2) :: BSj2=(/14,30/)     
      integer, dimension(3) :: BSj3=(/11,21,12/)     
      integer, dimension(4) :: BSj4=(/6,14,10,14/)     

      integer, dimension(1) :: BSk1=(/55/)     
      integer, dimension(2) :: BSk2=(/28,27/)     
      integer, dimension(3) :: BSk3=(/12,18,25/)     
      integer, dimension(4) :: BSk4=(/10,18,15,12/)     

      integer, allocatable :: A3(:,:,:)
      character(12), parameter :: tname='distrmix316 '

!dvm$ distribute
!dvm$* A3(WGT_BLOCK(WB1,7), WGT_BLOCK(WB2,8), MULT_BLOCK(m3))   
!dvm$ dynamic A3

      allocate (A3(AN1,AN2,AN3))

!dvm$ region
!dvm$ parallel (i,j,k) on A3(i,j,k)
      do i=1,AN1
         do j=1,AN2
             do k=1,AN3
                A3(i,j,k) = i*NL/10 + j*NL/100 + k  + 2       
             enddo                                     
          enddo
      enddo
!dvm$ end region

!!!!dvm$ redistribute A3(GEN_BLOCK(BSi),GEN_BLOCK(BSj),BLOCK)   

      select case(psize(1))
      case(1)
          select case(psize(2))
          case(1) 
!dvm$ redistribute A3(GEN_BLOCK(BSi1),GEN_BLOCK(BSj1),BLOCK)    
          case(2)
!dvm$ redistribute A3(GEN_BLOCK(BSi1),GEN_BLOCK(BSj2),BLOCK)    
          case(3)
!dvm$ redistribute A3(GEN_BLOCK(BSi1),GEN_BLOCK(BSj3),BLOCK)    
          case(4)
!dvm$ redistribute A3(GEN_BLOCK(BSi1),GEN_BLOCK(BSj4),BLOCK)    
          case default 
             goto 10
          endselect  

      case(2)
          select case(psize(2))
          case(1) 
!dvm$ redistribute A3(GEN_BLOCK(BSi2),GEN_BLOCK(BSj1),BLOCK)    
          case(2)
!dvm$ redistribute A3(GEN_BLOCK(BSi2),GEN_BLOCK(BSj2),BLOCK)    
          case(3)
!dvm$ redistribute A3(GEN_BLOCK(BSi2),GEN_BLOCK(BSj3),BLOCK)    
          case(4)
!dvm$ redistribute A3(GEN_BLOCK(BSi2),GEN_BLOCK(BSj4),BLOCK)    
          case default 
             goto 10
          endselect  

      case(3)
          select case(psize(2))
          case(1) 
!dvm$ redistribute A3(GEN_BLOCK(BSi3),GEN_BLOCK(BSj1),BLOCK)    
          case(2)
!dvm$ redistribute A3(GEN_BLOCK(BSi3),GEN_BLOCK(BSj2),BLOCK)    
          case(3)
!dvm$ redistribute A3(GEN_BLOCK(BSi3),GEN_BLOCK(BSj3),BLOCK)    
          case(4)
!dvm$ redistribute A3(GEN_BLOCK(BSi3),GEN_BLOCK(BSj4),BLOCK)    
          case default 
             goto 10
          endselect  

      case(4)
          select case(psize(2))
          case(1) 
!dvm$ redistribute A3(GEN_BLOCK(BSi4),GEN_BLOCK(BSj1),BLOCK)    
          case(2)
!dvm$ redistribute A3(GEN_BLOCK(BSi4),GEN_BLOCK(BSj2),BLOCK)    
          case(3)
!dvm$ redistribute A3(GEN_BLOCK(BSi4),GEN_BLOCK(BSj3),BLOCK)    
          case(4)
!dvm$ redistribute A3(GEN_BLOCK(BSi4),GEN_BLOCK(BSj4),BLOCK)    
          case default 
             goto 10
          endselect  

      case default 
         goto 10
      endselect  

!dvm$ region
!dvm$ parallel (i,j,k) on A3(i,j,k)
      do i=1,AN1
         do j=1,AN2
            do k=1,AN3
                A3(i,j,k) = A3(i,j,k) + 5
            enddo                                     
         enddo
      enddo
!dvm$ end region

!dvm$ redistribute A3(MULT_BLOCK(m1),WGT_BLOCK(WB1,7),MULT_BLOCK(m2))   

!dvm$ actual (erri)

!dvm$ region
!dvm$ parallel (i,j,k) on A3(i,j,k), reduction( min( erri ) )
      do i=1,AN1
         do j=1,AN2
            do k=1,AN3
               A3(i,j,k) = A3(i,j,k) - 2 
               if (A3(i,j,k) /= (i*NL/10 + j*NL/100 + k + 5)) then     
                    erri = min(erri,i*NL/10 + j*NL/100 + k)
               endif 
            enddo
          enddo
      enddo
!dvm$ end region
  
!dvm$ get_actual (erri)

      if (erri == ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 

 10   deallocate (A3)

      end subroutine distrmix316

C-----------------------------------------------------distrmix317
c 317  DISTRIBUTE arrA3[GEN_BLOCK][*][GEN_BLOCK]            
c              REDISTRIBUTE [MULT_BLOCK][WGT_BLOCK][*]     
c              REDISTRIBUTE [*][GEN_BLOCK][BLOCK]                                                         range 3 4 1
                                                          
      subroutine distrmix317 (psize)
      integer psize(3)

      integer, parameter :: AN1=12,AN2=16,AN3=12,NL=1000,ER=100000
      integer :: erri=ER,i,j,k

      integer, parameter :: m1 = 2, m2 = 4, m3 = 3 

      double precision, dimension(8) :: 
     >  WB=(/2., 1., 2.5, 3., 4., 3., 3.5, 4./)     

      integer, dimension(1) :: BSi1=(/12/)          
      integer, dimension(2) :: BSi2=(/4,8/)         
      integer, dimension(3) :: BSi3=(/2,7,3/)     
      integer, dimension(4) :: BSi4=(/2,3,4,3/)     

      integer, dimension(1) :: BSj1=(/16/)     
      integer, dimension(2) :: BSj2=(/11,5/)     
      integer, dimension(3) :: BSj3=(/8,2,6/)     
      integer, dimension(4) :: BSj4=(/1,3,4,8/)     !rem

      integer, dimension(1) :: BSk1=(/12/)     
      integer, dimension(2) :: BSk2=(/2,10/)     
      integer, dimension(3) :: BSk3=(/1,4,7/)     
      integer, dimension(4) :: BSk4=(/2,4,3,3/)     

      integer, allocatable ::  A3(:,:,:)
      character(12), parameter :: tname='distrmix317 '
               
!dvm$ distribute  :: A3
!dvm$ dynamic A3

      allocate (A3(AN1,AN2,AN3))

!!!!dvm$ redistribute A3(GEN_BLOCK(BSi),*,GEN_BLOCK(BSk))   
      select case(psize(1))
      case(1)
          select case(psize(2)) ! it's is true - psize(2)
          case(1) 
!dvm$ redistribute A3(GEN_BLOCK(BSi1),*,GEN_BLOCK(BSk1))    
          case(2)
!dvm$ redistribute A3(GEN_BLOCK(BSi1),*,GEN_BLOCK(BSk2))    
          case(3)
!dvm$ redistribute A3(GEN_BLOCK(BSi1),*,GEN_BLOCK(BSk3))    
          case(4)
!dvm$ redistribute A3(GEN_BLOCK(BSi1),*,GEN_BLOCK(BSk4))    
          case default 
             goto 10
          endselect  

      case(2)
          select case(psize(2)) ! it's is true - psize(2)
          case(1) 
!dvm$ redistribute A3(GEN_BLOCK(BSi2),*,GEN_BLOCK(BSk1))    
          case(2)
!dvm$ redistribute A3(GEN_BLOCK(BSi2),*,GEN_BLOCK(BSk2))    
          case(3)
!dvm$ redistribute A3(GEN_BLOCK(BSi2),*,GEN_BLOCK(BSk3))    
          case(4)
!dvm$ redistribute A3(GEN_BLOCK(BSi2),*,GEN_BLOCK(BSk4))    
          case default 
             goto 10
          endselect  

      case(3)
          select case(psize(2)) ! it's is true - psize(2)
          case(1) 
!dvm$ redistribute A3(GEN_BLOCK(BSi3),*,GEN_BLOCK(BSk1))    
          case(2)
!dvm$ redistribute A3(GEN_BLOCK(BSi3),*,GEN_BLOCK(BSk2))    
          case(3)
!dvm$ redistribute A3(GEN_BLOCK(BSi3),*,GEN_BLOCK(BSk3))    
          case(4)
!dvm$ redistribute A3(GEN_BLOCK(BSi3),*,GEN_BLOCK(BSk4))    
          case default 
             goto 10
          endselect  

      case(4)
          select case(psize(2)) ! it's is true - psize(2)
          case(1) 
!dvm$ redistribute A3(GEN_BLOCK(BSi4),*,GEN_BLOCK(BSk1))    
          case(2)
!dvm$ redistribute A3(GEN_BLOCK(BSi4),*,GEN_BLOCK(BSk2))    
          case(3)
!dvm$ redistribute A3(GEN_BLOCK(BSi4),*,GEN_BLOCK(BSk3))    
          case(4)
!dvm$ redistribute A3(GEN_BLOCK(BSi4),*,GEN_BLOCK(BSk4))    
          case default 
             goto 10
          endselect  

      case default 
         goto 10
      endselect  

!dvm$ region
!dvm$ parallel (i,j,k) on A3(i,j,k)
      do i=1,AN1
          do j=1,AN2
             do k=1,AN3
                  A3(i,j,k) = i*NL/10+j*NL/100+k
             enddo
          enddo
      enddo
!dvm$ end region

!dvm$ redistribute A3(MULT_BLOCK(m1),WGT_BLOCK(WB,8),*)   

!dvm$ region
!dvm$ parallel (i,j,k) on A3(i,j,k)
      do i=1,AN1
          do j=1,AN2
             do k=1,AN3
                 A3(i,j,k) = A3(i,j,k) * 2
             enddo
          enddo
      enddo
!dvm$ end region

!!!!dvm$*  redistribute A3(*,GEN_BLOCK(BSj),BLOCK)   

      select case (psize(1))                        !rem
      case(1)
!dvm$ redistribute A3(*,GEN_BLOCK(BSj1),BLOCK)    
      case(2)
!dvm$ redistribute A3(*,GEN_BLOCK(BSj2),BLOCK) 
      case(3)
!dvm$ redistribute A3(*,GEN_BLOCK(BSj3),BLOCK) 
      case(4)
!dvm$ redistribute A3(*,GEN_BLOCK(BSj4),BLOCK)  
      case default
         goto 10
      endselect

!dvm$ actual (erri)

!dvm$ region
!dvm$ parallel (i,j,k) on A3(i,j,k), reduction( min(erri) )
      do i=1,AN1
          do j=1,AN2
            do k=1,AN3
              A3(i,j,k) = A3(i,j,k) / 2
              if (A3(i,j,k) /= i*NL/10+j*NL/100+k) then
                  erri = min(erri,i*NL/10 + j*NL/100 + k)
              endif 
             enddo
           enddo
       enddo
!dvm$ end region
 
!dvm$ get_actual (erri)

      if (erri == ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 

 10   deallocate (A3)

      end subroutine distrmix317 

C ----------------------------------------------------distrmix318
c 318  DISTRIBUTE arrA3[GEN_BLOCK][GEN_BLOCK][GEN_BLOCK]          range 1 1 1 
c                REDISTRIBUTE [WGT_BLOCK][WGT_BLOCK][WGT_BLOCK]   range 1 4 3
c                REDISTRIBUTE [GEN_BLOCK][GEN_BLOCK][GEN_BLOCK]   range 2 3 2
c                                                                 range 3 1 4
c                                                                 range 4 2 2
      subroutine distrmix318 (psize)
      integer psize(3)

      integer, parameter :: AN1=8,AN2=8,AN3=8,NL=1000,ER=10000
      integer :: erri=ER,i,j,k

      double precision, dimension(7) :: 
     >             WB1=(/2.0, 1.2, 2.5, 1.4, 2.5, 1.3, 1./)     
      double precision, dimension(5) ::
     >             WB2=(/2., 1.3, 2., 1.0, 1.7/)     
      double precision, dimension(6) ::
     >             WB3=(/2., 3., 1.3, 2., 1.0, 1.7/)     

      integer, dimension(1) :: BSi111=(/8/)        !range 1 1 1  
      integer, dimension(1) :: BSj111=(/8/)     
      integer, dimension(1) :: BSk111=(/8/)     

      integer, dimension(1) :: BSi11=(/8/)         !range 1 4 3 
      integer, dimension(1) :: BSi12=(/8/)           
      integer, dimension(4) :: BSj11=(/3,2,2,1/)     
      integer, dimension(4) :: BSj12=(/4,2,1,1/)     
      integer, dimension(3) :: BSk11=(/4,3,1/)     
      integer, dimension(3) :: BSk12=(/2,4,2/)     

      integer, dimension(2) :: BSi21=(/6,2/)       !range 2 3 2  
      integer, dimension(2) :: BSi22=(/4,4/)         
      integer, dimension(3) :: BSj21=(/3,2,3/)     
      integer, dimension(3) :: BSj22=(/3,1,4/)     
      integer, dimension(2) :: BSk21=(/1,7/)     
      integer, dimension(2) :: BSk22=(/2,6/)     

      integer, dimension(3) :: BSi31=(/3,2,3/)     !range 3 1 4  
      integer, dimension(3) :: BSi32=(/4,2,2/)     
      integer, dimension(1) :: BSj31=(/8/)     
      integer, dimension(1) :: BSj32=(/8/)     
      integer, dimension(4) :: BSk31=(/1,3,2,2/)     
      integer, dimension(4) :: BSk32=(/1,1,4,2/)     

      integer, dimension(4) :: BSi41=(/3,2,1,2/)   !range 4 2 2  
      integer, dimension(4) :: BSi42=(/5,1,1,1/)   
      integer, dimension(2) :: BSj41=(/5,3/)     
      integer, dimension(2) :: BSj42=(/6,2/)     
      integer, dimension(2) :: BSk41=(/2,6/)     
      integer, dimension(2) :: BSk42=(/1,7/)     

      integer, allocatable :: A3(:,:,:)
      character(12), parameter :: tname='distrmix318 '

!dvm$ distribute :: A3   
!dvm$ dynamic A3

      allocate (A3(AN1,AN2,AN3))

!!!!dvm$ redistribute A3(GEN_BLOCK(BSi1),GEN_BLOCK(BSj1),GEN_BLOCK(BSk1))   

      select case(psize(1))
      case(1) 
        if  (psize(2) == 1 .and. psize(3) == 1) then
!dvm$ redistribute
!dvm$*    A3(GEN_BLOCK(BSi111),GEN_BLOCK(BSj111),GEN_BLOCK(BSk111))   
        else
!dvm$ redistribute
!dvm$*    A3(GEN_BLOCK(BSi11),GEN_BLOCK(BSj11),GEN_BLOCK(BSk11))   
        endif
      case(2)
!dvm$ redistribute
!dvm$*    A3(GEN_BLOCK(BSi21),GEN_BLOCK(BSj21),GEN_BLOCK(BSk21))   
      case(3)
!dvm$ redistribute
!dvm$*    A3(GEN_BLOCK(BSi31),GEN_BLOCK(BSj31),GEN_BLOCK(BSk31))   
      case(4)
!dvm$ redistribute
!dvm$*    A3(GEN_BLOCK(BSi41),GEN_BLOCK(BSj41),GEN_BLOCK(BSk41))   
      case default 
         goto 10
      endselect  

!dvm$ region
!dvm$ parallel (i,j,k) on A3(i,j,k)
      do i=1,AN1
          do j=1,AN2
             do k=1,AN3
                A3(i,j,k) = i*NL/10 + j*NL/100 + k    
             enddo
          enddo
      enddo
!dvm$ end region

!dvm$ redistribute
!dvm$*    A3(WGT_BLOCK(WB1,7),WGT_BLOCK(WB2,5),WGT_BLOCK(WB3,6))   

!dvm$ region
!dvm$ parallel (i,j,k) on A3(i,j,k)
      do i=1,AN1
          do j=1,AN2
             do k=1,AN3
                A3(i,j,k) = A3(i,j,k) + 1   
             enddo
          enddo
      enddo
!dvm$ end region

!!!!dvm$ redistribute A3(GEN_BLOCK(BSi2),GEN_BLOCK(BSj2),GEN_BLOCK(BSk2))   

      select case(psize(1))
      case(1) 
        if  (psize(2) == 1 .and. psize(3) == 1) then
!dvm$ redistribute
!dvm$*    A3(GEN_BLOCK(BSi111),GEN_BLOCK(BSj111),GEN_BLOCK(BSk111))   
        else
!dvm$ redistribute 
!dvm$*    A3(GEN_BLOCK(BSi12),GEN_BLOCK(BSj12),GEN_BLOCK(BSk12))   
        endif 
      case(2)
!dvm$ redistribute
!dvm$*    A3(GEN_BLOCK(BSi22),GEN_BLOCK(BSj22),GEN_BLOCK(BSk22))   
      case(3)
!dvm$ redistribute
!dvm$*    A3(GEN_BLOCK(BSi32),GEN_BLOCK(BSj32),GEN_BLOCK(BSk32))   
      case(4)
!dvm$ redistribute
!dvm$*    A3(GEN_BLOCK(BSi42),GEN_BLOCK(BSj42),GEN_BLOCK(BSk42))   
      case default 
         goto 10
      endselect  

!dvm$ actual (erri)

!dvm$ region
!dvm$ parallel (i,j,k) on A3(i,j,k), reduction(min(erri))
      do i=1,AN1
          do j=1,AN2
            do k=1,AN3
                if (A3(i,j,k) /= (i*NL/10 + j*NL/100 + k + 1)) then     
                    erri = min(erri,i*NL/10 + j*NL/100 + k)
                endif 
            enddo
          enddo
      enddo
!dvm$ end region
     
!dvm$ get_actual (erri)

      if (erri == ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 

 10   deallocate (A3)

      end subroutine distrmix318

C ----------------------------------------------------distrmix319
c  
c 319 DISTRIBUTE arrA3[WGT_BLOCK][WGT_BLOCK][WGT_BLOCK]           range 1 1 1 
c             REDISTRIBUTE [GEN_BLOCK][GEN_BLOCK][GEN_BLOCK]      range 1 2 1
c             REDISTRIBUTE arrA3[WGT_BLOCK][WGT_BLOCK][WGT_BLOCK] range 2 2 2
c                                                                 range 3 2 2
c                                                                 range 4 4 1
      subroutine distrmix319 (psize)
      integer psize(3)

      integer, parameter :: AN1=12,AN2=6,AN3=10,NL=1000,ER=10000
      integer :: erri=ER,i,j,k

      double precision, dimension(6) :: 
     >             WB1=(/2.0, 1.2, 2., 2.4, 2.3, 1.6/)     
      double precision, dimension(5) ::
     >             WB2=(/2.4, 1.8, 2., 1.0, 1.7/)     
      double precision, dimension(8) ::
     >             WB3=(/2., 3., 1.3, 2., 1.0, 1.7, 3., 4./)     

      integer, dimension(1) :: BSi111=(/8/)       !range 1 1 1  
      integer, dimension(1) :: BSj111=(/8/)     
      integer, dimension(1) :: BSk111=(/8/)     

      integer, dimension(1) :: BSi1=(/12/)        !range 1 2 1  
      integer, dimension(2) :: BSj1=(/5,1/)     
      integer, dimension(1) :: BSk1=(/10/)     !rem

      integer, dimension(2) :: BSi2=(/6,6/)       !range 2 2 2  
      integer, dimension(2) :: BSj2=(/4,2/)     
      integer, dimension(2) :: BSk2=(/3,7/)     

      integer, dimension(3) :: BSi3=(/5,2,5/)     !range 3 2 2  
      integer, dimension(2) :: BSj3=(/2,4/)     
      integer, dimension(2) :: BSk3=(/2,8/)     

      integer, dimension(4) :: BSi4=(/4,2,4,2/)   !range 4 4 1 !rem 
      integer, dimension(4) :: BSj4=(/1,1,2,2/)     
      integer, dimension(1) :: BSk4=(/10/)     

      integer, allocatable :: A3(:,:,:)
      character(12), parameter :: tname='distrmix319 '

!dvm$ distribute :: A3  
!dvm$ dynamic A3

      allocate (A3(AN1,AN2,AN3))

!dvm$ redistribute
!dvm$*    A3(WGT_BLOCK(WB1,6),WGT_BLOCK(WB2,5),WGT_BLOCK(WB3,8))   

       A3 = 10

!dvm$ actual (A3)

!dvm$ region
!dvm$ parallel (i,j,k) on A3(i,j,k)
      do i=1,AN1
          do j=1,AN2
             do k=1,AN3
                A3(i,j,k) = A3(i,j,k)+ i*NL/10 + j*NL/100 + k    
             enddo
          enddo
      enddo
!dvm$ end region

!!!!dvm$ redistribute A3(GEN_BLOCK(BSi),GEN_BLOCK(BSj),GEN_BLOCK(BSk))   

      select case(psize(1))
      case(1) 
        if  (psize(2) == 1 .and. psize(3) == 1) then
!dvm$ redistribute
!dvm$*    A3(GEN_BLOCK(BSi111),GEN_BLOCK(BSj111),GEN_BLOCK(BSk111))   
        else
!dvm$ redistribute 
!dvm$*    A3(GEN_BLOCK(BSi1),GEN_BLOCK(BSj1),GEN_BLOCK(BSk1))   
        endif
      case(2)
!dvm$ redistribute
!dvm$*    A3(GEN_BLOCK(BSi2),GEN_BLOCK(BSj2),GEN_BLOCK(BSk2))   
      case(3)
!dvm$ redistribute
!dvm$*    A3(GEN_BLOCK(BSi3),GEN_BLOCK(BSj3),GEN_BLOCK(BSk3))   
      case(4)
!dvm$ redistribute
!dvm$*    A3(GEN_BLOCK(BSi4),GEN_BLOCK(BSj4),GEN_BLOCK(BSk4))   
      case default 
         goto 10
      endselect  

!dvm$ region
!dvm$ parallel (i,j,k) on A3(i,j,k)
      do i=1,AN1
          do j=1,AN2
             do k=1,AN3
                A3(i,j,k) = A3(i,j,k) + 2   
             enddo
          enddo
      enddo
!dvm$ end region

!dvm$ redistribute
!dvm$*    A3(WGT_BLOCK(WB2,5),WGT_BLOCK(WB3,6),WGT_BLOCK(WB1,6))   

!dvm$ actual (erri)

!dvm$ region
!dvm$ parallel (i,j,k) on A3(i,j,k), reduction( min( erri ) )
      do i=1,AN1
          do j=1,AN2
            do k=1,AN3
                A3(i,j,k) = A3(i,j,k) - 12   
                if (A3(i,j,k) /= (i*NL/10 + j*NL/100 + k)) then     
                    erri = min(erri,i*NL/10 + j*NL/100 + k)
                endif 
            enddo
          enddo
      enddo
!dvm$ end region
     
!dvm$ get_actual (erri)

      if (erri == ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 

 10   deallocate (A3)

      end subroutine distrmix319

C ----------------------------------------------------distrmix320
c 320  DISTRIBUTE arrA3[BLOCK][GEN_BLOCK][GEN_BLOCK]  
c           REDISTRIBUTE arrB3[WGT_BLOCK][BLOCK][WGT_BLOCK]   

      subroutine distrmix320 (psize)
      integer psize(3)

      integer, parameter :: AN1=5,AN2=7,AN3=6,NL=1000,ER=10000
      integer :: erri=ER,i,j,k

      double precision, dimension(7) :: 
     >             WB1=(/2.0, 2.2, 3., 2.4, 2.3, 1.6, 0.5/)     
      double precision, dimension(6) ::
     >             WB2=(/2.4, 1.8, 3., 2.0, 1.7, 1./)     
      double precision, dimension(8) ::
     >             WB3=(/1., 3.5, 2.3, 2., 1.5, 1.7, 3., 2./)     

      integer, dimension(1) :: BSi1=(/5/)         
      integer, dimension(2) :: BSi2=(/1,4/)        
      integer, dimension(3) :: BSi3=(/1,2,2/)       
      integer, dimension(4) :: BSi4=(/2,1,1,1/)      

      integer, dimension(1) :: BSj1=(/7/)     
      integer, dimension(2) :: BSj2=(/3,4/)     
      integer, dimension(3) :: BSj3=(/2,4,1/)     
      integer, dimension(4) :: BSj4=(/1,2,1,3/)                

      integer, dimension(1) :: BSk1=(/6/)     
      integer, dimension(2) :: BSk2=(/3,3/)     
      integer, dimension(3) :: BSk3=(/2,3,1/)     
      integer, dimension(4) :: BSk4=(/3,2,0,1/)     


      integer, allocatable :: A3(:,:,:)
      character(12), parameter :: tname='distrmix320 '

!dvm$ distribute  :: A3
!dvm$ dynamic A3

      allocate (A3(AN1,AN2,AN3))

!!!!dvm$ redistribute A3(BLOCK,GEN_BLOCK(BSj),GEN_BLOCK(BSk))   

      select case(psize(2))
      case(1)
          select case(psize(3))
          case(1) 
!dvm$ redistribute A3(BLOCK,GEN_BLOCK(BSj1),GEN_BLOCK(BSk1))    
          case(2)
!dvm$ redistribute A3(BLOCK,GEN_BLOCK(BSj1),GEN_BLOCK(BSk2))    
          case(3)
!dvm$ redistribute A3(BLOCK,GEN_BLOCK(BSj1),GEN_BLOCK(BSk3))    
          case(4)
!dvm$ redistribute A3(BLOCK,GEN_BLOCK(BSj1),GEN_BLOCK(BSk4))    
          case default 
             goto 10
          endselect  

      case(2)
          select case(psize(3))
          case(1) 
!dvm$ redistribute A3(BLOCK,GEN_BLOCK(BSj2),GEN_BLOCK(BSk1))    
          case(2)
!dvm$ redistribute A3(BLOCK,GEN_BLOCK(BSj2),GEN_BLOCK(BSk2))    
          case(3)
!dvm$ redistribute A3(BLOCK,GEN_BLOCK(BSj2),GEN_BLOCK(BSk3))    
          case(4)
!dvm$ redistribute A3(BLOCK,GEN_BLOCK(BSj2),GEN_BLOCK(BSk4))    
          case default 
             goto 10
          endselect  

      case(3)
          select case(psize(3))
          case(1) 
!dvm$ redistribute A3(BLOCK,GEN_BLOCK(BSj3),GEN_BLOCK(BSk1))    
          case(2)
!dvm$ redistribute A3(BLOCK,GEN_BLOCK(BSj3),GEN_BLOCK(BSk2))    
          case(3)
!dvm$ redistribute A3(BLOCK,GEN_BLOCK(BSj3),GEN_BLOCK(BSk3))    
          case(4)
!dvm$ redistribute A3(BLOCK,GEN_BLOCK(BSj3),GEN_BLOCK(BSk4))    
          case default 
             goto 10
          endselect  

      case(4)
          select case(psize(3))
          case(1) 
!dvm$ redistribute A3(BLOCK,GEN_BLOCK(BSj4),GEN_BLOCK(BSk1))    
          case(2)
!dvm$ redistribute A3(BLOCK,GEN_BLOCK(BSj4),GEN_BLOCK(BSk2))    
          case(3)
!dvm$ redistribute A3(BLOCK,GEN_BLOCK(BSj4),GEN_BLOCK(BSk3))    
          case(4)
!dvm$ redistribute A3(BLOCK,GEN_BLOCK(BSj4),GEN_BLOCK(BSk4))   
          case default 
             goto 10
          endselect  

      case default 
         goto 10
      endselect  

      A3 = 5

!dvm$ actual (A3)

!dvm$ region
!dvm$ parallel (i,j,k) on A3(i,j,k)
      do i=1,AN1
          do j=1,AN2
             do k=1,AN3
                A3(i,j,k) = A3(i,j,k) + i*NL/10 + j*NL/100 + k    
             enddo
          enddo
      enddo
!dvm$ end region

!dvm$ redistribute
!dvm$*    A3(WGT_BLOCK(WB1,7),BLOCK,WGT_BLOCK(WB3,8))   

!dvm$ actual (erri)

!dvm$ region
!dvm$ parallel (i,j,k) on A3(i,j,k), reduction(min(erri))
      do i=1,AN1
          do j=1,AN2
            do k=1,AN3
                A3(i,j,k) = A3(i,j,k) - 5   
                if (A3(i,j,k) /= (i*NL/10 + j*NL/100 + k)) then     
                    erri = min(erri,i*NL/10 + j*NL/100 + k)
                endif 
            enddo
          enddo
      enddo
!dvm$ end region
     
!dvm$ get_actual (erri)

      if (erri == ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 

 10   deallocate (A3)

      end subroutine distrmix320

C ----------------------------------------------------distrmix321
c 321 DISTRIBUTE arrA3[WGT_BLOCK][BLOCK][WGT_BLOCK]         
c                REDISTRIBUTE [GEN_BLOCK][BLOCK][GEN_BLOCK] 
                                                           
      subroutine distrmix321 (psize)
      integer psize(3)

      integer, parameter :: AN1=16,AN2=16,AN3=16,NL=1000,ER=10000
      integer :: erri=ER,i,j,k

      double precision, dimension(6) :: 
     >             WB1=(/2.5, 3.6, 2.4, 2.3, 1.2, 0.5/)     
      double precision, dimension(5) ::
     >             WB2=(/1.4, 2.8, 3., 3.0, 1.1/)     
      double precision, dimension(7) ::
     >             WB3=(/1., 2.3, 2.2, 3.5, 1.7, 3., 2./)     

      integer, dimension(1) :: BSi1=(/16/)         
      integer, dimension(2) :: BSi2=(/11,5/)        
      integer, dimension(3) :: BSi3=(/1,12,3/)       
      integer, dimension(4) :: BSi4=(/6,4,5,1/)      

      integer, dimension(1) :: BSj1=(/16/)     
      integer, dimension(2) :: BSj2=(/3,13/)     
      integer, dimension(3) :: BSj3=(/2,4,10/)     
      integer, dimension(4) :: BSj4=(/5,1,7,3/)                

      integer, dimension(1) :: BSk1=(/16/)     
      integer, dimension(2) :: BSk2=(/10,6/)     
      integer, dimension(3) :: BSk3=(/2,8,6/)     
      integer, dimension(4) :: BSk4=(/3,2,10,1/)     

      integer A3(AN1,AN2,AN3)   ! static array
      character(12), parameter :: tname='distrmix321 '

!dvm$ distribute A3(WGT_BLOCK(WB1,6),BLOCK,WGT_BLOCK(WB3,7))   
!dvm$ dynamic A3

      A3 = 20 

!dvm$ actual (A3)

!dvm$ region
!dvm$ parallel (i,j,k) on A3(i,j,k)
      do i=1,AN1
          do j=1,AN2
             do k=1,AN3
                A3(i,j,k) =  A3(i,j,k)+ i*NL/10 + j*NL/100 + k   
             enddo
          enddo
      enddo
!dvm$ end region
   
!!!!dvm$ redistribute A3(GEN_BLOCK(BSi),BLOCK,GEN_BLOCK(BSk))   
   
      select case(psize(1))
      case(1)
          select case(psize(3))
          case(1) 
!dvm$ redistribute A3(GEN_BLOCK(BSi1),BLOCK,GEN_BLOCK(BSk1))    
          case(2)
!dvm$ redistribute A3(GEN_BLOCK(BSi1),BLOCK,GEN_BLOCK(BSk2))    
          case(3)
!dvm$ redistribute A3(GEN_BLOCK(BSi1),BLOCK,GEN_BLOCK(BSk3))    
          case(4)
!dvm$ redistribute A3(GEN_BLOCK(BSi1),BLOCK,GEN_BLOCK(BSk4))    
          case default 
             goto 10
          endselect  

      case(2)
          select case(psize(3))
          case(1) 
!dvm$ redistribute A3(GEN_BLOCK(BSi2),BLOCK,GEN_BLOCK(BSk1))    
          case(2)
!dvm$ redistribute A3(GEN_BLOCK(BSi2),BLOCK,GEN_BLOCK(BSk2))    
          case(3)
!dvm$ redistribute A3(GEN_BLOCK(BSi2),BLOCK,GEN_BLOCK(BSk3))    
          case(4)
!dvm$ redistribute A3(GEN_BLOCK(BSi2),BLOCK,GEN_BLOCK(BSk4))    
          case default 
             goto 10
          endselect  

      case(3)
          select case(psize(3))
          case(1) 
!dvm$ redistribute A3(GEN_BLOCK(BSi3),BLOCK,GEN_BLOCK(BSk1))    
          case(2)
!dvm$ redistribute A3(GEN_BLOCK(BSi3),BLOCK,GEN_BLOCK(BSk2))    
          case(3)
!dvm$ redistribute A3(GEN_BLOCK(BSi3),BLOCK,GEN_BLOCK(BSk3))    
          case(4)
!dvm$ redistribute A3(GEN_BLOCK(BSi3),BLOCK,GEN_BLOCK(BSk4))    
          case default 
             goto 10
          endselect  

      case(4)
          select case(psize(3))
          case(1) 
!dvm$ redistribute A3(GEN_BLOCK(BSi4),BLOCK,GEN_BLOCK(BSk1))    
          case(2)
!dvm$ redistribute A3(GEN_BLOCK(BSi4),BLOCK,GEN_BLOCK(BSk2))    
          case(3)
!dvm$ redistribute A3(GEN_BLOCK(BSi4),BLOCK,GEN_BLOCK(BSk3))    
          case(4)
!dvm$ redistribute A3(GEN_BLOCK(BSi4),BLOCK,GEN_BLOCK(BSk4))    
          case default 
             goto 10
          endselect  

      case default 
         goto 10
      endselect  

!dvm$ actual (erri)

!dvm$ region
!dvm$ parallel (i,j,k) on A3(i,j,k), reduction(min(erri))
      do i=1,AN1
          do j=1,AN2
            do k=1,AN3
                A3(i,j,k) = A3(i,j,k) - 20   
                if (A3(i,j,k) /= (i*NL/10 + j*NL/100 + k)) then     
                    erri = min(erri,i*NL/10 + j*NL/100 + k)
                endif 
            enddo
          enddo
      enddo
!dvm$ end region
     
!dvm$ get_actual (erri)

      if (erri == ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 

 10   continue

      end subroutine distrmix321

C ----------------------------------------------------distrmix322
c 322  DISTRIBUTE arrA3[GEN_BLOCK][BLOCK][GEN_BLOCK]    
c                REDISTRIBUTE [BLOCK][WGT_BLOCK][BLOCK] 
                                                       
      subroutine distrmix322 (psize)
      integer psize(3)

      integer, parameter :: AN1=24,AN2=16,AN3=8,NL=1000,ER=10000
      integer :: erri=ER,i,j,k

      double precision, dimension(5) :: 
     >             WB1=(/ 3.2, 2.4, 2.0, 1.0, 2.5/)     
      double precision, dimension(4) ::
     >             WB2=(/2.1, 2.5, 3., 1.1/)     
      double precision, dimension(6) ::
     >             WB3=(/2.3, 2.0, 3.5, 1.5, 3., 2./)     

      integer, dimension(1) :: BSi1=(/24/)         
      integer, dimension(2) :: BSi2=(/11,13/)        
      integer, dimension(3) :: BSi3=(/10,12,2/)       
      integer, dimension(4) :: BSi4=(/6,14,3,1/)      

      integer, dimension(1) :: BSj1=(/16/)     
      integer, dimension(2) :: BSj2=(/12,4/)     
      integer, dimension(3) :: BSj3=(/3,7,6/)     
      integer, dimension(4) :: BSj4=(/4,2,6,4/)                

      integer, dimension(1) :: BSk1=(/8/)     
      integer, dimension(2) :: BSk2=(/2,6/)     
      integer, dimension(3) :: BSk3=(/3,1,4/)     
      integer, dimension(4) :: BSk4=(/4,2,1,1/)                

      integer, allocatable :: A3(:,:,:)
      character(12), parameter :: tname='distrmix322 '

!dvm$ distribute :: A3
!dvm$ dynamic A3

      allocate (A3(AN1,AN2,AN3))

!!!!dvm$ redistribute A3(GEN_BLOCK(BSi),BLOCK,GEN_BLOCK(BSk))   

      select case(psize(1))
      case(1)
          select case(psize(3))
          case(1) 
!dvm$ redistribute A3(GEN_BLOCK(BSi1),BLOCK,GEN_BLOCK(BSk1))    
          case(2)
!dvm$ redistribute A3(GEN_BLOCK(BSi1),BLOCK,GEN_BLOCK(BSk2))    
          case(3)
!dvm$ redistribute A3(GEN_BLOCK(BSi1),BLOCK,GEN_BLOCK(BSk3))    
          case(4)
!dvm$ redistribute A3(GEN_BLOCK(BSi1),BLOCK,GEN_BLOCK(BSk4))    
          case default 
             goto 10
          endselect  

      case(2)
          select case(psize(3))
          case(1) 
!dvm$ redistribute A3(GEN_BLOCK(BSi2),BLOCK,GEN_BLOCK(BSk1))    
          case(2)
!dvm$ redistribute A3(GEN_BLOCK(BSi2),BLOCK,GEN_BLOCK(BSk2))    
          case(3)
!dvm$ redistribute A3(GEN_BLOCK(BSi2),BLOCK,GEN_BLOCK(BSk3))    
          case(4)
!dvm$ redistribute A3(GEN_BLOCK(BSi2),BLOCK,GEN_BLOCK(BSk4))    
          case default 
             goto 10
          endselect  

      case(3)
          select case(psize(3))
          case(1) 
!dvm$ redistribute A3(GEN_BLOCK(BSi3),BLOCK,GEN_BLOCK(BSk1))    
          case(2)
!dvm$ redistribute A3(GEN_BLOCK(BSi3),BLOCK,GEN_BLOCK(BSk2))    
          case(3)
!dvm$ redistribute A3(GEN_BLOCK(BSi3),BLOCK,GEN_BLOCK(BSk3))    
          case(4)
!dvm$ redistribute A3(GEN_BLOCK(BSi3),BLOCK,GEN_BLOCK(BSk4))    
          case default 
             goto 10
          endselect  

      case(4)
          select case(psize(3))
          case(1) 
!dvm$ redistribute A3(GEN_BLOCK(BSi4),BLOCK,GEN_BLOCK(BSk1))    
          case(2)
!dvm$ redistribute A3(GEN_BLOCK(BSi4),BLOCK,GEN_BLOCK(BSk2))    
          case(3)
!dvm$ redistribute A3(GEN_BLOCK(BSi4),BLOCK,GEN_BLOCK(BSk3))    
          case(4)
!dvm$ redistribute A3(GEN_BLOCK(BSi4),BLOCK,GEN_BLOCK(BSk4))    
          case default 
             goto 10
          endselect  

      case default 
         goto 10
      endselect  

      A3 = 15  

!dvm$ actual (A3)

!dvm$ region
!dvm$ parallel (i,j,k) on A3(i,j,k)
      do i=1,AN1
          do j=1,AN2
             do k=1,AN3
                A3(i,j,k) = A3(i,j,k) + i*NL/10 + j*NL/100 + k    
             enddo
          enddo
      enddo
!dvm$ end region

!dvm$ redistribute  A3(BLOCK,WGT_BLOCK(WB2,4),BLOCK)   

!dvm$ actual (erri)

!dvm$ region
!dvm$ parallel (i,j,k) on A3(i,j,k), reduction(min(erri))
      do i=1,AN1
          do j=1,AN2
            do k=1,AN3
                if (A3(i,j,k) /= (i*NL/10 + j*NL/100 + k + 15)) then     
                    erri = min(erri,i*NL/10 + j*NL/100 + k)
                endif 
            enddo
          enddo
      enddo
!dvm$ end region
     
!dvm$ get_actual (erri)

      if (erri == ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 

 10   deallocate (A3)

      end subroutine distrmix322

C ----------------------------------------------------distrmix323
c 323 DISTRIBUTE arrA3[BLOCK][WGT_BLOCK][BLOCK]             
c                REDISTRIBUTE [GEN_BLOCK][GEN_BLOCK][*] 
                                                           
      subroutine distrmix323 (psize)
      integer psize(3)

      integer, parameter :: AN1=8,AN2=11,AN3=11,NL=1000,ER=10000
      integer :: erri=ER,i,j,k

      double precision, dimension(7) :: 
     >             WB1=(/ 3.2, 2.4, 1., 2., 2.0, 1.0, 2.5/)     
      double precision, dimension(6) ::
     >             WB2=(/3.1, 2.5, 4., 2.1, 2., 2./)
      double precision, dimension(6) ::
     >             WB3=(/1.2, 3.0, 2.4, 1.0, 3., 2.5/)     

      integer, dimension(1) :: BSi1=(/8/)         
      integer, dimension(2) :: BSi2=(/2,6/)        
      integer, dimension(3) :: BSi3=(/1,3,4/)       
      integer, dimension(4) :: BSi4=(/3,2,1,2/)      

      integer, dimension(1) :: BSj1=(/11/)     
      integer, dimension(2) :: BSj2=(/3,8/)     
      integer, dimension(3) :: BSj3=(/1,7,3/)     
      integer, dimension(4) :: BSj4=(/5,3,1,2/)                

      integer, dimension(1) :: BSk1=(/11/)     
      integer, dimension(2) :: BSk2=(/1,10/)     
      integer, dimension(3) :: BSk3=(/3,4,4/)     
      integer, dimension(4) :: BSk4=(/4,2,2,3/)                

      integer :: A3(AN1,AN2,AN3)
      character(12), parameter :: tname='distrmix323 '

!dvm$ distribute A3(BLOCK,WGT_BLOCK(WB2,6),BLOCK)   
!dvm$ dynamic A3

!dvm$ region
!dvm$ parallel (i,j,k) on A3(i,j,k)
      do i=1,AN1
          do j=1,AN2
             do k=1,AN3
                A3(i,j,k) = i*NL/10 + j*NL/100 + k    
             enddo
          enddo
      enddo
!dvm$ end region

!!!!dvm$ redistribute A3(GEN_BLOCK(BSi),GEN_BLOCK(BSj),*)   
   
      select case(psize(1))
      case(1)
          select case(psize(2))
          case(1) 
!dvm$ redistribute A3(GEN_BLOCK(BSi1),GEN_BLOCK(BSj1),*)    
          case(2)
!dvm$ redistribute A3(GEN_BLOCK(BSi1),GEN_BLOCK(BSj2),*)    
          case(3)
!dvm$ redistribute A3(GEN_BLOCK(BSi1),GEN_BLOCK(BSj3),*)    
          case(4)
!dvm$ redistribute A3(GEN_BLOCK(BSi1),GEN_BLOCK(BSj4),*)    
          case default 
              goto 10
          endselect  

      case(2)
          select case(psize(2))
          case(1) 
!dvm$ redistribute A3(GEN_BLOCK(BSi2),GEN_BLOCK(BSj1),*)    
          case(2)
!dvm$ redistribute A3(GEN_BLOCK(BSi2),GEN_BLOCK(BSj2),*)    
          case(3)
!dvm$ redistribute A3(GEN_BLOCK(BSi2),GEN_BLOCK(BSj3),*)    
          case(4)
!dvm$ redistribute A3(GEN_BLOCK(BSi2),GEN_BLOCK(BSj4),*)    
          case default 
             goto 10
          endselect  

      case(3)
          select case(psize(2))
          case(1) 
!dvm$ redistribute A3(GEN_BLOCK(BSi3),GEN_BLOCK(BSj1),*)    
          case(2)
!dvm$ redistribute A3(GEN_BLOCK(BSi3),GEN_BLOCK(BSj2),*)    
          case(3)
!dvm$ redistribute A3(GEN_BLOCK(BSi3),GEN_BLOCK(BSj3),*)    
          case(4)
!dvm$ redistribute A3(GEN_BLOCK(BSi3),GEN_BLOCK(BSj4),*)    
          case default 
             goto 10
          endselect  

      case(4)
          select case(psize(2))
          case(1) 
!dvm$ redistribute A3(GEN_BLOCK(BSi4),GEN_BLOCK(BSj1),*)    
          case(2)
!dvm$ redistribute A3(GEN_BLOCK(BSi4),GEN_BLOCK(BSj2),*)    
          case(3)
!dvm$ redistribute A3(GEN_BLOCK(BSi4),GEN_BLOCK(BSj3),*)    
          case(4)
!dvm$ redistribute A3(GEN_BLOCK(BSi4),GEN_BLOCK(BSj4),*)    
          case default 
             goto 10
          endselect  

      case default 
         goto 10
      endselect  

!dvm$ actual (erri)

!dvm$ region
!dvm$ parallel (i,j,k) on A3(i,j,k), reduction(min(erri))
      do i=1,AN1
          do j=1,AN2
            do k=1,AN3
                if (A3(i,j,k) /= (i*NL/10 + j*NL/100 + k)) then     
                    erri = min(erri,i*NL/10 + j*NL/100 + k)
                endif 
            enddo
          enddo
      enddo
!dvm$ end region
     
!dvm$ get_actual (erri)

      if (erri == ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 

 10   continue

      end subroutine distrmix323

C ----------------------------------------------------distrmix324
c 324 DISTRIBUTE arrA3[GEN_BLOCK][GEN_BLOCK][*]         
c                REDISTRIBUTE [*][WGT_BLOCK][WGT_BLOCK] 
c                REDISTRIBUTE [GEN_BLOCK][*][GEN_BLOCK] 
c                                                       
      subroutine distrmix324 (psize)
      integer psize(3)

      integer, parameter :: AN1=12,AN2=12,AN3=21,NL=1000,ER=10000
      integer :: erri=ER,i,j,k

      double precision, dimension(7) :: 
     >             WB1=(/2.0, 1.2, 2.5, 1.4, 2.5, 1.3, 1./)     
      double precision, dimension(5) ::
     >             WB2=(/2., 1.3, 2., 1.0, 1.7/)     
      double precision, dimension(6) ::
     >             WB3=(/2., 3., 1.3, 2., 1.0, 1.7/)     

      integer, dimension(1) :: BSi11=(/12/)      
      integer, dimension(1) :: BSi12=(/12/)            
      integer, dimension(2) :: BSi21=(/8,4/)    
      integer, dimension(2) :: BSi22=(/2,10/)         
      integer, dimension(3) :: BSi31=(/1,6,5/)    
      integer, dimension(3) :: BSi32=(/4,6,2/)     
      integer, dimension(4) :: BSi41=(/3,2,4,3/)  
      integer, dimension(4) :: BSi42=(/4,2,2,4/)   

      integer, dimension(1) :: BSj11=(/12/) 
      integer, dimension(1) :: BSj12=(/12/)     
      integer, dimension(2) :: BSj21=(/6,6/)     
      integer, dimension(2) :: BSj22=(/1,11/)     
      integer, dimension(3) :: BSj31=(/8,2,2/)     
      integer, dimension(3) :: BSj32=(/1,10,1/)     
      integer, dimension(4) :: BSj41=(/2,5,3,2/)     
      integer, dimension(4) :: BSj42=(/2,8,1,1/)     

      integer, dimension(1) :: BSk11=(/21/)     
      integer, dimension(1) :: BSk12=(/21/)     
      integer, dimension(2) :: BSk21=(/11,10/)     
      integer, dimension(2) :: BSk22=(/7,14/)     
      integer, dimension(3) :: BSk31=(/1,5,15/)     
      integer, dimension(3) :: BSk32=(/4,6,11/)     
      integer, dimension(4) :: BSk41=(/1,2,10,8/)     
      integer, dimension(4) :: BSk42=(/12,4,2,3/)     

      integer, allocatable :: A3(:,:,:)
      character(*), parameter :: tname='distrmix324 '

!dvm$ distribute :: A3   
!dvm$ dynamic A3

      allocate (A3(AN1,AN2,AN3))

!!!!dvm$ redistribute A3(GEN_BLOCK(BSi1),GEN_BLOCK(BSj1),*)   

      select case(psize(1))
      case(1)
          select case(psize(2))
          case(1) 
!dvm$ redistribute A3(GEN_BLOCK(BSi11),GEN_BLOCK(BSj11),*)    
          case(2)
!dvm$ redistribute A3(GEN_BLOCK(BSi11),GEN_BLOCK(BSj21),*)    
          case(3)
!dvm$ redistribute A3(GEN_BLOCK(BSi11),GEN_BLOCK(BSj31),*)    
          case(4)
!dvm$ redistribute A3(GEN_BLOCK(BSi11),GEN_BLOCK(BSj41),*)    
          case default 
             goto 10
          endselect  

      case(2)
          select case(psize(2))
          case(1) 
!dvm$ redistribute A3(GEN_BLOCK(BSi21),GEN_BLOCK(BSj11),*)    
          case(2)
!dvm$ redistribute A3(GEN_BLOCK(BSi21),GEN_BLOCK(BSj21),*)    
          case(3)
!dvm$ redistribute A3(GEN_BLOCK(BSi21),GEN_BLOCK(BSj31),*)    
          case(4)
!dvm$ redistribute A3(GEN_BLOCK(BSi21),GEN_BLOCK(BSj41),*)    
          case default 
             goto 10
          endselect  

      case(3)
          select case(psize(2))
          case(1) 
!dvm$ redistribute A3(GEN_BLOCK(BSi31),GEN_BLOCK(BSj11),*)    
          case(2)
!dvm$ redistribute A3(GEN_BLOCK(BSi31),GEN_BLOCK(BSj21),*)    
          case(3)
!dvm$ redistribute A3(GEN_BLOCK(BSi31),GEN_BLOCK(BSj31),*)    
          case(4)
!dvm$ redistribute A3(GEN_BLOCK(BSi31),GEN_BLOCK(BSj41),*)    
          case default 
             goto 10
          endselect  

      case(4)
          select case(psize(2))
          case(1) 
!dvm$ redistribute A3(GEN_BLOCK(BSi41),GEN_BLOCK(BSj11),*)    
          case(2)
!dvm$ redistribute A3(GEN_BLOCK(BSi41),GEN_BLOCK(BSj21),*)    
          case(3)
!dvm$ redistribute A3(GEN_BLOCK(BSi41),GEN_BLOCK(BSj31),*)    
          case(4)
!dvm$ redistribute A3(GEN_BLOCK(BSi41),GEN_BLOCK(BSj41),*)    
          case default 
             goto 10
          endselect  

      case default 
         goto 10
      endselect  

      A3 = 1 

!dvm$ actual (A3)

!dvm$ region
!dvm$ parallel (i,j,k) on A3(i,j,k)
      do i=1,AN1
          do j=1,AN2
             do k=1,AN3
                A3(i,j,k) = A3(i,j,k)+i*NL/10 + j*NL/100 + k    
             enddo
          enddo
      enddo
!dvm$ end region

!dvm$ redistribute
!dvm$*    A3(*, WGT_BLOCK(WB2,5),WGT_BLOCK(WB3,6))   

!dvm$ region
!dvm$ parallel (i,j,k) on A3(i,j,k)
      do i=1,AN1
          do j=1,AN2
             do k=1,AN3
                A3(i,j,k) = A3(i,j,k) * 2   
             enddo
          enddo
      enddo
!dvm$ end region 

!!!!dvm$ redistribute A3(GEN_BLOCK(BSi2),*,GEN_BLOCK(BSk2))   

      select case(psize(1))
      case(1)
          select case(psize(2)) ! it's true - psize(2)
          case(1) 
!dvm$ redistribute A3(GEN_BLOCK(BSi12),*,GEN_BLOCK(BSk12))    
          case(2)
!dvm$ redistribute A3(GEN_BLOCK(BSi12),*,GEN_BLOCK(BSk22))    
          case(3)
!dvm$ redistribute A3(GEN_BLOCK(BSi12),*,GEN_BLOCK(BSk32))    
          case(4)                                            
!dvm$ redistribute A3(GEN_BLOCK(BSi12),*,GEN_BLOCK(BSk42))
          case default
             goto 10
          endselect 
      case(2)
          select case(psize(2)) ! it's true - psize(2)
          case(1) 
!dvm$ redistribute A3(GEN_BLOCK(BSi22),*,GEN_BLOCK(BSk12))    
          case(2)
!dvm$ redistribute A3(GEN_BLOCK(BSi22),*,GEN_BLOCK(BSk22))    
          case(3)
!dvm$ redistribute A3(GEN_BLOCK(BSi22),*,GEN_BLOCK(BSk32))    
          case(4)
!dvm$ redistribute A3(GEN_BLOCK(BSi22),*,GEN_BLOCK(BSk42))    
          case default 
             goto 10
          endselect  

      case(3)
          select case(psize(2)) ! it's true - psize(2)
          case(1) 
!dvm$ redistribute A3(GEN_BLOCK(BSi32),*,GEN_BLOCK(BSk12))    
          case(2)
!dvm$ redistribute A3(GEN_BLOCK(BSi32),*,GEN_BLOCK(BSk22))    
          case(3)
!dvm$ redistribute A3(GEN_BLOCK(BSi32),*,GEN_BLOCK(BSk32))    
          case(4)
!dvm$ redistribute A3(GEN_BLOCK(BSi32),*,GEN_BLOCK(BSk42))    
          case default 
             goto 10
          endselect  

      case(4)
          select case(psize(2)) ! it's true - psize(2)
          case(1) 
!dvm$ redistribute A3(GEN_BLOCK(BSi42),*,GEN_BLOCK(BSk12))    
          case(2)
!dvm$ redistribute A3(GEN_BLOCK(BSi42),*,GEN_BLOCK(BSk22))    
          case(3)
!dvm$ redistribute A3(GEN_BLOCK(BSi42),*,GEN_BLOCK(BSk32))    
          case(4)
!dvm$ redistribute A3(GEN_BLOCK(BSi42),*,GEN_BLOCK(BSk42))    
          case default 
             goto 10
          endselect  

      case default 
         goto 10
      endselect  

!dvm$ actual (erri)

!dvm$ region
!dvm$ parallel (i,j,k) on A3(i,j,k), reduction(min(erri))
      do i=1,AN1
          do j=1,AN2
            do k=1,AN3
                A3(i,j,k) = A3(i,j,k) / 2   
                if (A3(i,j,k) /= (i*NL/10 + j*NL/100 + k + 1)) then     
                    erri = min(erri,i*NL/10 + j*NL/100 + k)
                endif 
            enddo
          enddo
      enddo
!dvm$ end region
     
!dvm$ get_actual (erri)

      if (erri == ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 

 10   deallocate (A3)

      end subroutine distrmix324

C ----------------------------------------------------distrmix325
c 325  DISTRIBUTE arrA3 [*][GEN_BLOCK][GEN_BLOCK]                
c                REDISTRIBUTE [*][WGT_BLOCK][*]                 
c                REDISTRIBUTE [GEN_BLOCK][BLOCK][GEN_BLOCK]  
c                                                               
c                                                               
      subroutine distrmix325 (psize)
      integer psize(3)

      integer, parameter :: AN1=7,AN2=6,AN3=7,NL=1000,ER=10000
      integer :: erri=ER,i,j,k

      double precision, dimension(10) ::
     >          WB2=(/2.0, 1.2, 2.5, 1.0, 2.5, 1.3, 1., 3., 2., 1./)     

      integer, dimension(1) :: BSi11=(/7/)      
      integer, dimension(1) :: BSi12=(/7/)            
      integer, dimension(2) :: BSi21=(/3,4/)    
      integer, dimension(2) :: BSi22=(/2,5/)         
      integer, dimension(3) :: BSi31=(/1,6,0/)    
      integer, dimension(3) :: BSi32=(/4,2,1/)     
      integer, dimension(4) :: BSi41=(/3,2,1,1/)  
      integer, dimension(4) :: BSi42=(/2,1,2,2/)   

      integer, dimension(1) :: BSj11=(/6/) 
      integer, dimension(1) :: BSj12=(/6/)     
      integer, dimension(2) :: BSj21=(/2,4/)     
      integer, dimension(2) :: BSj22=(/0,6/)     
      integer, dimension(3) :: BSj31=(/2,2,2/)     
      integer, dimension(3) :: BSj32=(/1,3,2/)     
      integer, dimension(4) :: BSj41=(/2,1,1,2/)     
      integer, dimension(4) :: BSj42=(/3,0,2,1/)     

      integer, dimension(1) :: BSk11=(/7/)     
      integer, dimension(1) :: BSk12=(/7/)     
      integer, dimension(2) :: BSk21=(/3,4/)     
      integer, dimension(2) :: BSk22=(/6,1/)     
      integer, dimension(3) :: BSk31=(/1,5,1/)     
      integer, dimension(3) :: BSk32=(/4,2,1/)     
      integer, dimension(4) :: BSk41=(/2,0,3,2/)     
      integer, dimension(4) :: BSk42=(/2,4,0,1/)     

      integer, allocatable :: A3(:,:,:)
      character(12), parameter :: tname='distrmix325 '

!dvm$ distribute :: A3
!dvm$ dynamic A3

      allocate (A3(AN1,AN2,AN3))

!!!!dvm$ redistribute A3(*,GEN_BLOCK(BSj1),GEN_BLOCK(BSk1))   

                select case(psize(1))     ! it's true - psize(1)
      case(1)
         select case(psize(2))  ! it's true - psize(2)
         case(1) 
!dvm$ redistribute
!dvm$*    A3(*,GEN_BLOCK(BSj11),GEN_BLOCK(BSk11))   
         case(2) 
!dvm$ redistribute
!dvm$*    A3(*,GEN_BLOCK(BSj11),GEN_BLOCK(BSk21))   
         case(3)
!dvm$ redistribute
!dvm$*    A3(*,GEN_BLOCK(BSj11),GEN_BLOCK(BSk31))   
         case(4)
!dvm$ redistribute
!dvm$*    A3(*,GEN_BLOCK(BSj11),GEN_BLOCK(BSk41))   
         case default 
            goto 10
         endselect  

      case(2)
         select case(psize(2))  ! it's true - psize(2)
         case(1) 
!dvm$ redistribute
!dvm$*    A3(*,GEN_BLOCK(BSj21),GEN_BLOCK(BSk11))   
         case(2)
!dvm$ redistribute
!dvm$*    A3(*,GEN_BLOCK(BSj21),GEN_BLOCK(BSk21))   
         case(3)
!dvm$ redistribute
!dvm$*    A3(*,GEN_BLOCK(BSj21),GEN_BLOCK(BSk31))   
         case(4)
!dvm$ redistribute
!dvm$*    A3(*,GEN_BLOCK(BSj21),GEN_BLOCK(BSk41))   
         case default 
            goto 10               
         endselect  

      case (3)
         select case(psize(2))  ! it's true - psize(2)
         case(1) 
!dvm$ redistribute
!dvm$*    A3(*,GEN_BLOCK(BSj31),GEN_BLOCK(BSk11))   
         case(2)
!dvm$ redistribute
!dvm$*    A3(*,GEN_BLOCK(BSj31),GEN_BLOCK(BSk21))   
         case(3)
!dvm$ redistribute
!dvm$*    A3(*,GEN_BLOCK(BSj31),GEN_BLOCK(BSk31))   
         case(4)
!dvm$ redistribute
!dvm$*    A3(*,GEN_BLOCK(BSj31),GEN_BLOCK(BSk41))   
         case default 
            goto 10
         endselect  

      case (4)
         select case(psize(2))  ! it's true - psize(2)
         case(1) 
!dvm$ redistribute
!dvm$*    A3(*,GEN_BLOCK(BSj41),GEN_BLOCK(BSk11))   
         case(2)
!dvm$ redistribute
!dvm$*    A3(*,GEN_BLOCK(BSj41),GEN_BLOCK(BSk21))   
         case(3)
!dvm$ redistribute
!dvm$*    A3(*,GEN_BLOCK(BSj41),GEN_BLOCK(BSk31))   
         case(4)
!dvm$ redistribute
!dvm$*    A3(*,GEN_BLOCK(BSj41),GEN_BLOCK(BSk41))   
         case default 
            goto 10
         endselect  

      case default 
         goto 10
      endselect  

!dvm$ region  
!dvm$ parallel (i,j,k) on A3(i,j,k)
      do i=1,AN1
          do j=1,AN2
             do k=1,AN3
                A3(i,j,k) = i*NL/10 + j*NL/100 + k + 5  
             enddo
          enddo
      enddo
!dvm$ end region   



!dvm$ region
!dvm$ parallel (i,j,k) on A3(i,j,k)
      do i=1,AN1
          do j=1,AN2
             do k=1,AN3
                A3(i,j,k) = i*NL/10 + j*NL/100 + k + 5  
             enddo
          enddo
      enddo
!dvm$ end region

!dvm$ redistribute A3(*, WGT_BLOCK(WB2,10), *)   

!dvm$ region
!dvm$ parallel (i,j,k) on A3(i,j,k)
      do i=1,AN1
          do j=1,AN2
             do k=1,AN3
                A3(i,j,k) = A3(i,j,k) + 5   
             enddo
          enddo
      enddo
!dvm$ end region

!!!!dvm$ redistribute A3(GEN_BLOCK(BSi2),BLOCK,GEN_BLOCK(BSk2))   

      select case(psize(1))
      case(1)
          select case(psize(3))
          case(1) 
!dvm$ redistribute A3(GEN_BLOCK(BSi12),BLOCK,GEN_BLOCK(BSk12))    
          case(2)
!dvm$ redistribute A3(GEN_BLOCK(BSi12),BLOCK,GEN_BLOCK(BSk22))    
          case (3)
!dvm$ redistribute A3(GEN_BLOCK(BSi12),BLOCK,GEN_BLOCK(BSk32))    
          case(4)
!dvm$ redistribute A3(GEN_BLOCK(BSi12),BLOCK,GEN_BLOCK(BSk42))    
          case default 
             goto 10
          endselect  

      case(2)
          select case(psize(3))
          case(1) 
!dvm$ redistribute A3(GEN_BLOCK(BSi22),BLOCK,GEN_BLOCK(BSk12))    
          case(2)
!dvm$ redistribute A3(GEN_BLOCK(BSi22),BLOCK,GEN_BLOCK(BSk22))    
          case (3)
!dvm$ redistribute A3(GEN_BLOCK(BSi22),BLOCK,GEN_BLOCK(BSk32))    
          case(4)
!dvm$ redistribute A3(GEN_BLOCK(BSi22),BLOCK,GEN_BLOCK(BSk42))    
          case default 
             goto 10
          endselect  

      case (3)
          select case(psize(3))
          case(1) 
!dvm$ redistribute A3(GEN_BLOCK(BSi32),BLOCK,GEN_BLOCK(BSk12))    
          case(2)
!dvm$ redistribute A3(GEN_BLOCK(BSi32),BLOCK,GEN_BLOCK(BSk22))    
          case (3)
!dvm$ redistribute A3(GEN_BLOCK(BSi32),BLOCK,GEN_BLOCK(BSk32))    
          case(4)
!dvm$ redistribute A3(GEN_BLOCK(BSi32),BLOCK,GEN_BLOCK(BSk42))    
          case default 
             goto 10
          endselect  

      case(4)
          select case(psize(3))
          case(1) 
!dvm$ redistribute A3(GEN_BLOCK(BSi42),BLOCK,GEN_BLOCK(BSk12))    
          case(2)
!dvm$ redistribute A3(GEN_BLOCK(BSi42),BLOCK,GEN_BLOCK(BSk22))    
          case (3)
!dvm$ redistribute A3(GEN_BLOCK(BSi42),BLOCK,GEN_BLOCK(BSk32))    
          case(4)
!dvm$ redistribute A3(GEN_BLOCK(BSi42),BLOCK,GEN_BLOCK(BSk42))    
          case default 
             goto 10
          endselect  

      case default 
         goto 10
      endselect  


!dvm$ actual (erri)

!dvm$ region
!dvm$ parallel (i,j,k) on A3(i,j,k), reduction(min(erri))
      do i=1,AN1
          do j=1,AN2
            do k=1,AN3
                A3(i,j,k) = A3(i,j,k)    
                if (A3(i,j,k) /= (i*NL/10 + j*NL/100 + k + 10)) then     
                    erri = min(erri,i*NL/10 + j*NL/100 + k)
                endif 
            enddo
          enddo
      enddo
!dvm$ end region
     
!dvm$ get_actual (erri)

      if (erri == ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 

 10   deallocate (A3)

      end subroutine distrmix325


C -------------------------------------------------

      subroutine ansyes(name)
      character(*) name
      print *,name,'  -  complete'
      end

      subroutine ansno(name)
      character(*) name
      print *,name,'  -  ***error'
      end