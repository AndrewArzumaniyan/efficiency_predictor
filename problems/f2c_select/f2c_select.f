      program SELECT_SIMPLE

c    TESTING convert statement SELECT .       

      print *,'===START OF F2C_SELECT ========================'
C --------------------------------------------------
c   normal select
      call select_with_default
c   only default node select
      call select_only_default
c   select without default node
      call select_without_default
c   select with interval
      call select_interval
c   select with multi interval
      call select_multi_interval
c   select with multi select
      call select_multi_select
      print *,'=== END OF F2C_SELECT ========================= '    
      end

C ----------------------------------------------------select11
      subroutine select_with_default
      integer, parameter :: AN1=8, ER=10000    
      character*22 tname
      integer, allocatable :: A1(:)
      integer, allocatable :: B1(:)
      integer erri,i
               
!dvm$ distribute A1(BLOCK)    

      tname='select_with_default'
      allocate (A1(AN1))
      allocate (B1(AN1))
      
      do i=1,AN1
            B1(i) =i     
      enddo
!dvm$ region
!dvm$ parallel (i) on A1(i)
      do i=1,AN1
            A1(i) =i     
      enddo

!dvm$ parallel (i) on A1(i), reduction( min( erri ) )
!dvm$*, private(ia)
      do i=1, AN1
        ia = A1(i)
        select case( MOD( A1(i), 4 ) )
        case(0)
            ia = ia + 4
        case(1)
            ia = ia+3
        case(2)
            ia = ia+2
        case default
            ia = ia+1
        end select  
        A1(i) = ia*2+3
      enddo
!dvm$ end region   

      do i=1, AN1
        ia = B1(i)
        select case( MOD( B1(i), 4 ) )
        case(0)
            ia = ia + 4
        case(1)
            ia = ia+3
        case(2)
            ia = ia+2
        case default
            ia = ia+1
        end select    
        B1(i) = ia*2+3
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
C ----------------------------------------------------select12
      subroutine select_only_default
      integer, parameter :: AN1=8, ER=10000    
      character*22 tname
      integer, allocatable :: A1(:)
      integer, allocatable :: B1(:)
      integer erri,i
               
!dvm$ distribute A1(BLOCK)    

      tname='select_only_default'
      allocate (A1(AN1))
      allocate (B1(AN1))
      
      do i=1,AN1
            B1(i) =i     
      enddo
!dvm$ region
!dvm$ parallel (i) on A1(i)
      do i=1,AN1
            A1(i) =i     
      enddo

!dvm$ parallel (i) on A1(i), reduction( min( erri ) )
!dvm$*, private(ia)
      do i=1, AN1
        ia = A1(i)
        select case( MOD( A1(i), 4) )
          case default
            ia = ia*A1(i)-15
        end select    
        A1(i) = ia
      enddo
!dvm$ end region   

      do i=1, AN1
        ia = B1(i)
        select case( MOD( B1(i), 4) )
          case default
            ia = ia*B1(i)-15
        end select    
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
      

      erri= ER
      if (erri .eq.ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      deallocate (A1)
      deallocate (B1)

      end

C ----------------------------------------------------select13
      subroutine select_without_default
      integer, parameter :: AN1=8, ER=10000    
      character*22 tname
      integer, allocatable :: A1(:)
      integer, allocatable :: B1(:)
      integer erri,i
               
!dvm$ distribute A1(BLOCK)    

      tname='select_without_default'
      allocate (A1(AN1))
      allocate (B1(AN1))
      
      do i=1,AN1
            B1(i) =i     
      enddo
!dvm$ region
!dvm$ parallel (i) on A1(i)
      do i=1,AN1
            A1(i) =i     
      enddo

!dvm$ parallel (i) on A1(i), reduction( min( erri ) )
!dvm$*, private(ia)
      do i=1, AN1
        ia = A1(i)
        select case( MOD( A1(i), 4 ) )
        case(0)
            ia = ia + 4
        case(1)
            ia = ia*2+3
        case(2)
            ia = ia*3-7
        end select  
        A1(i) = ia
      enddo
!dvm$ end region   

      do i=1, AN1
        ia = B1(i)
        select case( MOD( B1(i), 4 ) )
        case(0)
            ia = ia + 4
        case(1)
            ia = ia*2+3
        case(2)
            ia = ia*3-7
        end select  
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
      

      erri= ER
      if (erri .eq.ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      deallocate (A1)
      deallocate (B1)

      end


C ----------------------------------------------------select14
      subroutine select_interval
      integer, parameter :: AN1=8, ER=10000    
      character*22 tname
      integer, allocatable :: A1(:)
      integer, allocatable :: B1(:)
      integer erri,i
               
!dvm$ distribute A1(BLOCK)    

      tname='select_interval'
      allocate (A1(AN1))
      allocate (B1(AN1))
      
      do i=1,AN1
            B1(i) =i     
      enddo
!dvm$ region 
!dvm$ parallel (i) on A1(i)
      do i=1,AN1
            A1(i) =i     
      enddo

!dvm$ parallel (i) on A1(i), reduction( min( erri ) )
!dvm$*, private(ia)
      do i=1, AN1
        ia = A1(i)
        select case( MOD( A1(i), 20 ) )
        case(:7)
            ia = ia + 4
        case(9:13)
            ia = ia*2+3
        case(16:)
            ia = ia*3-7
        case default
            ia = A1(i)*1/8 +ia*A1(i)-ia
        end select  
        A1(i) = ia
      enddo
!dvm$ end region   

      do i=1, AN1
        ia = B1(i)
        select case( MOD( B1(i), 20 ) )
        case(:7)
            ia = ia + 4
        case(9:13)
            ia = ia*2+3
        case(16:)
            ia = ia*3-7
        case default
            ia = B1(i)*1/8 +ia*B1(i)-ia
        end select  
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
      

      erri= ER
      if (erri .eq.ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      deallocate (A1)
      deallocate (B1)

      end

C ----------------------------------------------------select15
      subroutine select_multi_interval
      integer, parameter :: AN1=8, ER=10000    
      character*22 tname
      integer, allocatable :: A1(:)
      integer, allocatable :: B1(:)
      integer erri,i
               
!dvm$ distribute A1(BLOCK)    

      tname='select_multi_interval'
      allocate (A1(AN1))
      allocate (B1(AN1))
      
      do i=1,AN1
            B1(i) =i     
      enddo
!dvm$ region 
!dvm$ parallel (i) on A1(i)
      do i=1,AN1
            A1(i) =i     
      enddo

!dvm$ parallel (i) on A1(i), reduction( min( erri ) )
!dvm$*, private(ia)
      do i=1, AN1
        ia = A1(i)
        select case( MOD( A1(i), 30 ) )
        case(:4)
            ia = ia + 4
        case(9:13, 20:24, 5)
            ia = ia*2+3
        case(7, 17:19, 26: )
            ia = ia*3-7
        case default
            ia = A1(i)*1/8 +ia*A1(i)-ia
        end select  
        A1(i) = ia
      enddo
!dvm$ end region   

      do i=1, AN1
        ia = B1(i)
        select case( MOD( B1(i), 30 ) )
        case(:4)
            ia = ia + 4
        case(9:13, 20:24, 5)
            ia = ia*2+3
        case(7, 17:19, 26: )
            ia = ia*3-7
        case default
            ia = B1(i)*1/8 +ia*B1(i)-ia
        end select  
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
      

      erri= ER
      if (erri .eq.ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      deallocate (A1)
      deallocate (B1)

      end
C ----------------------------------------------------select16
      subroutine select_multi_select
      integer, parameter :: AN1=8, ER=10000    
      character*22 tname
      integer, allocatable :: A1(:)
      integer, allocatable :: B1(:)
      integer erri,i
               
!dvm$ distribute A1(BLOCK)    

      tname='select_multi_interval'
      allocate (A1(AN1))
      allocate (B1(AN1))
      
      do i=1,AN1
            B1(i) =i     
      enddo
!dvm$ region
!dvm$ parallel (i) on A1(i)
      do i=1,AN1
            A1(i) =i     
      enddo

!dvm$ parallel (i) on A1(i), reduction( min( erri ) )
!dvm$*, private(ia)
      do i=1, AN1
        ia = A1(i)
        select case( MOD( A1(i), 30 ) )
        case(:4)
            ia = ia + 4
        case(9:13, 20:24, 5)
            ia = ia*2+3
            select case(ia + min(A1(i) +7, A1(i)*A1(i)*1/4-19 ))
            case(:10)
                ia = max(ia, 19) + 7
            case default
                ia = ia/2 -9
            case(17)
                ia = ia+1
            case(20:)
                ia = A1(i)-7
            end select
        case(7, 17:19, 26: )
            ia = ia*3-7
        case default
            ia = A1(i)*1/8 +ia*A1(i)-ia
        end select  
        A1(i) = ia
      enddo
!dvm$ end region   

      do i=1, AN1
        ia = B1(i)
         select case( MOD( B1(i), 30 ) )
        case(:4)
            ia = ia + 4
        case(9:13, 20:24, 5)
            ia = ia*2+3
            select case(ia + min(B1(i) +7, A1(i)*B1(i)*1/4-19 ))
            case(:10)
                ia = max(ia, 19) + 7
            case default
                ia = ia/2 -9
            case(17)
                ia = ia+1
            case(20:)
                ia = B1(i)-7
            end select
        case(7, 17:19, 26: )
            ia = ia*3-7
        case default
            ia = B1(i)*1/8 +ia*B1(i)-ia
        end select  
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
      

      erri= ER
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
      character*22 name
      print *,name,'  -  complete'
      end
      subroutine ansno(name)
      character*22 name
      print *,name,'  -  ***error'
      end
