      program DO1

c    TESTING convert statement DO  .       

      print *,'===START OF F2C_DO========================'
C --------------------------------------------------
c   do with enddo
      call do_enddo
c   do with label (continue)
      call do_continue
c   do with label (last stmt)
      call do_without_end
c   check iterator value after DO
      call do_value_iter
c   check iterator value in same step (+3)
      call do_with_same_step1
c   check iterator value in same step (-2)
      call do_with_same_step2
c   multi do
      call do_multi
c   cycle stmt
      call do_cycle_stmt_1
      call do_cycle_stmt_2
c   exit stmt
      call do_exit_stmt
c   do while with var-expr
      call do_while_true
c   do while const-expr
      call do_while_expr
    

      print *,'=== END OF F2C_DO ========================= '    
      end

C ----------------------------------------------------do1
      subroutine do_enddo
      integer, parameter :: AN1=256, ER=10000    
      character*18 tname
      integer, allocatable :: A1(:)
      integer, allocatable :: B1(:)
      integer erri,i, ia
!dvm$ distribute A1(BLOCK)
      tname='do_enddo'
	  
	  
    
      allocate (A1(AN1))
      allocate (B1(AN1))
      
      erri= ER
      do i=1,AN1
            B1(i) =i     
      enddo

!dvm$ region
!dvm$ parallel (i) on A1(i)
      do i=1,AN1
            A1(i) =i     
      enddo

!dvm$ parallel (i) on A1(i)
!dvm$*, private(ia)
      do i=1, AN1
            do ia = 1, i
                 A1(i) = A1(i) + ia + (i-5)
            enddo
      enddo
      
!dvm$ end region   
      do i=1, AN1
            do ia = 1, i
                B1(i) = B1(i) + ia + (i-5)
            enddo
      enddo
      erri= ER
     

!dvm$ get_actual(A1)
!dvm$ parallel (i) on A1(i), reduction( min( erri ) )
      do i=1,AN1
            if ( abs(A1(i) - B1(i) ) .lt. 0.001 ) then         
            else
               erri = min(erri,i)
            endif     
      enddo
      

     
      if (erri .eq.ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
	  deallocate (A1)
	  deallocate (B1)


      end
C ----------------------------------------------------do12
      subroutine do_continue
      integer, parameter :: AN1=256, ER=10000    
      character*18 tname
      integer, allocatable :: A1(:)
      integer, allocatable :: B1(:)
      integer erri,i, ia
!dvm$ distribute A1(BLOCK)
      tname='do_continue'
	  
	  
    
      allocate (A1(AN1))
      allocate (B1(AN1))
      erri= ER
      do i=1,AN1
            B1(i) =i     
      enddo

!dvm$ region
!dvm$ parallel (i) on A1(i)
      do i=1,AN1
            A1(i) =i     
      enddo

!dvm$ parallel (i) on A1(i)
!dvm$*, private(ia)
      do i=1, AN1
            do 101, ia = 1, i
                 A1(i) = A1(i) + ia + (i-5)
101          continue
      enddo
      
!dvm$ end region   
      do i=1, AN1
            do 201, ia = 1, i
                B1(i) = B1(i) + ia + (i-5)
201          continue
      enddo
      erri= ER
!dvm$ get_actual(A1)
!dvm$ parallel (i) on A1(i), reduction( min( erri ) )
      do i=1,AN1
            if ( abs(A1(i) - B1(i) ) .lt. 0.001 ) then         
            else
               erri = min(erri,i)
            endif     
      enddo
      

      erri= ER
      if (erri .eq.ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
	  deallocate (A1)
	  deallocate (B1)


      end

C ----------------------------------------------------do13
      subroutine do_without_end
      integer, parameter :: AN1=256, ER=10000    
      character*18 tname
      integer, allocatable :: A1(:)
      integer, allocatable :: B1(:)
      integer erri,i, ia
!dvm$ distribute A1(BLOCK)
      tname='do_without_end'
	  
	  
    
      allocate (A1(AN1))
      allocate (B1(AN1))
      erri= ER
      do i=1,AN1
            B1(i) =i     
      enddo

!dvm$ region
!dvm$ parallel (i) on A1(i)
      do i=1,AN1
            A1(i) =i     
      enddo

!dvm$ parallel (i) on A1(i)
!dvm$*, private(ia)
      do i=1, AN1
            do 102, ia = 1, i
102            A1(i) = A1(i) + ia + (i-5)
         
      enddo
      
!dvm$ end region   
      do i=1, AN1
            do 202, ia = 1, i
202             B1(i) = B1(i) + ia + (i-5)
          
      enddo
      erri= ER
!dvm$ get_actual(A1)
!dvm$ parallel (i) on A1(i), reduction( min( erri ) )
      do i=1,AN1
            if ( abs(A1(i) - B1(i) ) .lt. 0.001 ) then         
            else
               erri = min(erri,i)
            endif     
      enddo
      

     
      if (erri .eq.ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
	  deallocate (A1)
	  deallocate (B1)


      end


C ----------------------------------------------------do14
      subroutine do_value_iter
      integer, parameter :: AN1=256, ER=10000    
      character*18 tname
      integer, allocatable :: A1(:)
      integer, allocatable :: B1(:)
      integer erri,i, ia
!dvm$ distribute A1(BLOCK)
      tname='do_value_iter'
	  
	  
    
      allocate (A1(AN1))
      allocate (B1(AN1))
      erri= ER
      do i=1,AN1
            B1(i) =i     
      enddo

!dvm$ region
!dvm$ parallel (i) on A1(i)
      do i=1,AN1
            A1(i) =i     
      enddo


!dvm$ parallel (i) on A1(i)
!dvm$*, private(ia)
      do i=1, AN1
            do ia = 1, i*2-5
                 A1(i) = A1(i) + ia + (i-5)
            enddo
            A1(i) = ia
            
      enddo
      
!dvm$ end region   
      do i=1, AN1
            do ia = 1, i*2-5
                B1(i) = B1(i) + ia + (i-5)
            enddo
            B1(i) = ia
      enddo
      erri= ER
!dvm$ get_actual(A1)
!dvm$ parallel (i) on A1(i), reduction( min( erri ) )
      do i=1,AN1
            if ( abs(A1(i) - B1(i) ) .lt. 0.001 ) then         
            else
               erri = min(erri,i)
            endif     
      enddo
      

     
      if (erri .eq.ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
	  deallocate (A1)
	  deallocate (B1)


      end

C ----------------------------------------------------do15
      subroutine do_with_same_step1
      integer, parameter :: AN1=256, ER=10000    
      character*18 tname
      integer, allocatable :: A1(:)
      integer, allocatable :: B1(:)
      integer erri,i, ia
!dvm$ distribute A1(BLOCK)
      tname='do_with_same_step1'
	  
	  
    
      allocate (A1(AN1))
      allocate (B1(AN1))
      erri= ER
      do i=1,AN1
            B1(i) =i     
      enddo

!dvm$ region
!dvm$ parallel (i) on A1(i)
      do i=1,AN1
            A1(i) =i     
      enddo


!dvm$ parallel (i) on A1(i)
!dvm$*, private(ia)
      do i=1, AN1
            do ia = 1, i*2-5, 3
                 A1(i) = A1(i) + ia + (i-5)
            enddo
            
      enddo
      
!dvm$ end region   
      do i=1, AN1
            do ia = 1, i*2-5, 3
                B1(i) = B1(i) + ia + (i-5)
            enddo
      enddo
      erri= ER
!dvm$ get_actual(A1)
!dvm$ parallel (i) on A1(i), reduction( min( erri ) )
      do i=1,AN1
            if ( abs(A1(i) - B1(i) ) .lt. 0.001 ) then         
            else
               erri = min(erri,i)
            endif     
      enddo
      

     
      if (erri .eq.ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
	  deallocate (A1)
	  deallocate (B1)


      end
C ----------------------------------------------------do16
      subroutine do_with_same_step2
      integer, parameter :: AN1=256, ER=10000    
      character*18 tname
      integer, allocatable :: A1(:)
      integer, allocatable :: B1(:)
      integer erri,i, ia
!dvm$ distribute A1(BLOCK)
      tname='do_with_same_step2'
	  
	  
    
      allocate (A1(AN1))
      allocate (B1(AN1))
      erri= ER
      do i=1,AN1
            B1(i) =i     
      enddo

!dvm$ region
!dvm$ parallel (i) on A1(i)
      do i=1,AN1
            A1(i) =i     
      enddo


!dvm$ parallel (i) on A1(i)
!dvm$*, private(ia)
      do i=1, AN1
            do ia = i*2-5, 1, -2
                 A1(i) = A1(i) + ia + (i-5)
            enddo
            
      enddo
      
!dvm$ end region   
      do i=1, AN1
            do ia = i*2-5, 1, -2
                B1(i) = B1(i) + ia + (i-5)
            enddo
      enddo
      erri= ER
!dvm$ get_actual(A1)
!dvm$ parallel (i) on A1(i), reduction( min( erri ) )
      do i=1,AN1
            if ( abs(A1(i) - B1(i) ) .lt. 0.001 ) then         
            else
               erri = min(erri,i)
            endif     
      enddo
      

     
      if (erri .eq.ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
	  deallocate (A1)
	  deallocate (B1)


      end
C ----------------------------------------------------do17
      subroutine do_multi
      integer, parameter :: AN1=256, ER=10000    
      character*18 tname
      integer, allocatable :: A1(:)
      integer, allocatable :: B1(:)
      integer erri,i, ia
!dvm$ distribute A1(BLOCK)
      tname='do_multi'
	  
	  
    
      allocate (A1(AN1))
      allocate (B1(AN1))
      erri= ER
      do i=1,AN1
            B1(i) =i     
      enddo

!dvm$ region
!dvm$ parallel (i) on A1(i)
      do i=1,AN1
            A1(i) =i     
      enddo


!dvm$ parallel (i) on A1(i)
!dvm$*, private(ia,j,n)
      do i=1, AN1
        n = 0
        do 107, ia = 1, A1(i)
            do 107, j = ia, A1(i)
107             n = n+1         
        A1(i) = n + j - 2*ia
      enddo
!dvm$ end region

      do i=1, AN1
        n = 0
        do 207, ia = 1, B1(i)
            do 207, j = ia, B1(i)
207             n = n+1         
        B1(i) = n + j - 2*ia
      enddo

      erri= ER
!dvm$ get_actual(A1)
!dvm$ parallel (i) on A1(i), reduction( min( erri ) )
      do i=1,AN1
            if ( abs(A1(i) - B1(i) ) .lt. 0.001 ) then         
            else
               erri = min(erri,i)
            endif     
      enddo   

     
      if (erri .eq.ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
	  deallocate (A1)
	  deallocate (B1)

      end
C ----------------------------------------------------do18
      subroutine do_cycle_stmt_1
      integer, parameter :: AN1=256, ER=10000    
      character*18 tname
      integer, allocatable :: A1(:)
      integer, allocatable :: B1(:)
      integer erri,i, ia
!dvm$ distribute A1(BLOCK)
      tname='do_cycle_stmt_1'
	  
	  
    
      allocate (A1(AN1))
      allocate (B1(AN1))
      erri= ER
      do i=1,AN1
            B1(i) =i     
      enddo

!dvm$ region
!dvm$ parallel (i) on A1(i)
      do i=1,AN1
            A1(i) =i     
      enddo

!dvm$ parallel (i) on A1(i)
!dvm$*, private(ia)
      do i=1, AN1
            do ia = 1, i*2-5, 2
                 if(mod(A1(i),2) .eq.0) cycle
                 A1(i) = A1(i) + ia  + (i-5)
            enddo            
      enddo
      
!dvm$ end region   
      do i=1, AN1
            do ia = 1, i*2-5, 2
                if(mod(B1(i),2) .eq.0) cycle
                B1(i) = B1(i) + ia + (i-5)
            enddo
      enddo
      erri= ER
!dvm$ get_actual(A1)
!dvm$ parallel (i) on A1(i), reduction( min( erri ) )
      do i=1,AN1
            if ( abs(A1(i) - B1(i) ) .lt. 0.001 ) then         
            else
               erri = min(erri,i)
            endif     
      enddo
      
     
      if (erri .eq.ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
	  deallocate (A1)
	  deallocate (B1)


      end
C ----------------------------------------------------do19
      subroutine do_cycle_stmt_2
      integer, parameter :: AN1=256, ER=10000    
      character*18 tname
      integer, allocatable :: A1(:)
      integer, allocatable :: B1(:)
      integer erri,i, ia
!dvm$ distribute A1(BLOCK)
      tname='do_cycle_stmt_2'
	  
	  
    
      allocate (A1(AN1))
      allocate (B1(AN1))
      erri= ER
      do i=1,AN1
            B1(i) =i     
      enddo

!dvm$ region
!dvm$ parallel (i) on A1(i)
      do i=1,AN1
            A1(i) =i     
      enddo

!dvm$ parallel (i) on A1(i)
!dvm$*, private(ia)
      do i=1, AN1
            if(A1(i) .gt. 5)  cycle
            do ia = 1, i-200
                 A1(i) = A1(i)+ia+(i-5)
            enddo            
      enddo      
!dvm$ end region   
      do i=1, AN1
            if(B1(i) .gt. 5)  cycle
            do ia = 1, i-200
                 B1(i) = B1(i)+ia+(i-5)
            enddo
      enddo
      erri= ER
!dvm$ get_actual(A1)
!dvm$ parallel (i) on A1(i), reduction( min( erri ) )
      do i=1,AN1
            if ( abs(A1(i) - B1(i) ) .lt. 0.001 ) then         
            else
               erri = min(erri,i)
            endif     
      enddo
      
     
      if (erri .eq.ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
	  deallocate (A1)
	  deallocate (B1)


      end

C ----------------------------------------------------do20
      subroutine do_exit_stmt
      integer, parameter :: AN1=256, ER=10000    
      character*18 tname
      integer, allocatable :: A1(:)
      integer, allocatable :: B1(:)
      integer erri,i, ia
!dvm$ distribute A1(BLOCK)
      tname='do_exit_stmt'
	  
	  
    
      allocate (A1(AN1))
      allocate (B1(AN1))
      erri= ER
      do i=1,AN1
            B1(i) =i     
      enddo

!dvm$ region
!dvm$ parallel (i) on A1(i)
      do i=1,AN1
            A1(i) =i     
      enddo



!dvm$ parallel (i) on A1(i)
!dvm$*, private(ia,j,n)
      do i=1, AN1
        n = 0
        do ia = 1, A1(i)
            j = 1
            do
                n = n+1
                if(j .gt. ia) then 
                    n = n-1
                    exit
                endif
                j = j+1
            enddo        
        enddo    
        A1(i) = n+A1(i)+2*j-3*ia 
      enddo
!dvm$ end region   


      do i=1, AN1
        n = 0
        do ia = 1, B1(i)
            j = 1
            do
                n = n+1
                if(j .gt. ia) then 
                    n = n-1
                    exit
                endif
                j = j+1
            enddo        
        enddo    
        B1(i) = n+B1(i)+2*j-3*ia
      enddo
      erri= ER
!dvm$ get_actual(A1)
!dvm$ parallel (i) on A1(i), reduction( min( erri ) )
      do i=1,AN1
            if ( abs(A1(i) - B1(i) ) .lt. 0.001 ) then         
            else
               erri = min(erri,i)
            endif     
      enddo  
      
      
      if (erri .eq.ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
	  deallocate (A1)
	  deallocate (B1)

      end
C ----------------------------------------------------do21
      subroutine do_while_true
      integer, parameter :: AN1=256, ER=10000    
      character*18 tname
      integer, allocatable :: A1(:)
      integer, allocatable :: B1(:)
      integer erri,i, ia
!dvm$ distribute A1(BLOCK)
      tname='do_while_true'
	  
	  
    
      allocate (A1(AN1))
      allocate (B1(AN1))
      erri= ER
      do i=1,AN1
            B1(i) =i     
      enddo

!dvm$ region
!dvm$ parallel (i) on A1(i)
      do i=1,AN1
            A1(i) =i     
      enddo


!dvm$ parallel (i) on A1(i)
      do i=1, AN1
            do while(.true.)
                A1(i) = A1(i) + i
                if(A1(i) .gt. 2*A1(i) .or. i .gt. A1(i) / 3 - 5) exit
            enddo
      enddo
      
!dvm$ end region   
      do i=1, AN1
            do while(.true.)
                B1(i) = B1(i) + i
                if(B1(i) .gt. 2*B1(i) .or. i .gt. B1(i) / 3 - 5) exit
            enddo
      enddo
      erri= ER
!dvm$ get_actual(A1)
!dvm$ parallel (i) on A1(i), reduction( min( erri ) )
      do i=1,AN1
            if ( abs(A1(i) - B1(i) ) .lt. 0.001 ) then         
            else
               erri = min(erri,i)
            endif     
      enddo
      

     
      if (erri .eq.ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
	  deallocate (A1)
	  deallocate (B1)


      end
C ----------------------------------------------------do22
      subroutine do_while_expr
      integer, parameter :: AN1=256, ER=10000    
      character*18 tname
      integer, allocatable :: A1(:)
      integer, allocatable :: B1(:)
      integer erri,i, ia
!dvm$ distribute A1(BLOCK)
      tname='do_while_expr'
	  
	  
    
      allocate (A1(AN1))
      allocate (B1(AN1))
      erri= ER
      do i=1,AN1
            B1(i) =i     
      enddo

!dvm$ region
!dvm$ parallel (i) on A1(i)
      do i=1,AN1
            A1(i) =i     
      enddo


!dvm$ parallel (i) on A1(i)
      do i=1, AN1
            do while(A1(i)*3 -40 .lt. A1(i) + 15)
                A1(i) = A1(i) + i
                if(A1(i) .gt. 2*A1(i) .or. i .gt. A1(i) / 3 - 5) exit
            enddo
      enddo
      
!dvm$ end region   
      do i=1, AN1
            do while(B1(i)*3 -40 .lt. B1(i) + 15)
                B1(i) = B1(i) + i
                if(B1(i) .gt. 2*B1(i) .or. i .gt. B1(i) / 3 - 5) exit
            enddo
      enddo
      erri= ER
!dvm$ get_actual(A1)
!dvm$ parallel (i) on A1(i), reduction( min( erri ) )
      do i=1,AN1
            if ( abs(A1(i) - B1(i) ) .lt. 0.001 ) then         
            else
               erri = min(erri,i)
            endif     
      enddo
      

     
      if (erri .eq.ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
	  deallocate (A1)
	  deallocate (B1)



      end
            
C -------------------------------------------------

      subroutine ansyes(name)
      character*18 name
      print *,name,'  -  complete'
      end
      subroutine ansno(name)
      character*18 name
      print *,name,'  -  ***error'
      end
