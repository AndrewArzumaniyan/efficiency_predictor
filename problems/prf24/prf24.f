program prf24
    !TESTING OF THE PREFETCH DIRECTIVE.

    print *, '===START OF PRF24========================'

    call prf2401
    call prf2402
    call prf2403

    print *, '===END OF PRF24=========================='
end

subroutine prf2401
    integer, parameter :: N = 4, M = 4, NL = 1000, NIT = 3
    integer, allocatable :: A( :, : ), B( :, : ), C( :, : ), D( :, : )
    character * 7 :: tname = 'PRF2401'

    !dvm$ distribute B( block, * )
    !dvm$ align ( :, : ) with B( :, : ) :: A, D

    !dvm$ remote_group GR1
    !dvm$ remote_group GR2
    !dvm$ remote_group GR3

    allocate( B( N, M ), A( N, M ), C( N, M ), D( N, M ) )
    call serial2( C, N, M, NL )

    !dvm$ parallel ( i, j ) on A( i, j )
    do i = 1, N
        do j = 1, M
            A( i, j ) = NL + i + j
            B( i, j ) = NL + i + j
            D( i, j ) = NL + i + j
        enddo
    enddo

    do it = 1, NIT
        !dvm$ prefetch GR1
        !dvm$ prefetch GR2
        !dvm$ prefetch GR3

        !dvm$ remote_access( GR1:A( N / 2, M / 2 ) )
        ib1 = A( N / 2, M / 2 )

        !dvm$ remote_access( GR1:B( N / 2, M ) )
        ib2 = B( N / 2, M )

        !dvm$ remote_access( GR2:D( N, M / 2 ) )
        ib3 = D( N, M / 2 )

        !dvm$ remote_access( GR3:D( N / 2, 1 ) )
        ib4 = D( N / 2, 1 )

        if ( ( ib1 .eq. C( N / 2, M / 2 ) ) .and. ( ib2 .eq. C( N / 2, M ) ) .and. &
             ( ib3 .eq. C( N, M / 2 ) ) .and. ( ib4 .eq. C( N / 2, 1 ) ) ) then
          call ansyes( tname )
        else
          call ansno( tname )
        endif
        if ( it .eq. 2 ) cycle
        !dvm$ reset GR1
        !dvm$ reset GR2
        !dvm$ reset GR3
    enddo
    deallocate( A, B, C, D )
end

subroutine prf2402
    integer, parameter ::  N = 4, M = 4, NL = 1000, NIT = 3
    integer, allocatable :: A( :, : ), B( :, : ), C( :, : ), D( :, : )
    integer, allocatable :: A1( :, : )
    character * 7 :: tname = 'PRF2402'

    !dvm$ distribute B( *, block )
    !dvm$ align( :, : ) with B( :, : ) :: A, A1

    !dvm$ remote_group GR1
    !dvm$ remote_group GR2
    !dvm$ remote_group GR3

    allocate( B( N, M ), A( N, M ), C( N, M ), D( N, M ), A1( N, M ) )
    call serial2( C, N, M, NL )

    !dvm$ parallel ( i, j ) on A( i, j )
    do i = 1, N
        do j = 1, M
            A( i, j ) = NL + i + j
            A1( i, j ) = NL + i + j
        enddo
    enddo

    do it = 1, NIT
        !dvm$ prefetch GR1
        !dvm$ prefetch GR2
        !dvm$ prefetch GR3

        isumc1 = 0
        isuma1 = 0
        !dvm$ remote_access ( GR1:A( :, M / 2 ) )
        do i = 1, N
            D( i, M / 2 ) = A( i, M / 2 )
            isumc1 = isumc1 + C( i, M / 2 )
            isuma1 = isuma1 + D( i, M / 2 )
        enddo

        isumc2 = 0
        isuma2 = 0
        !dvm$ remote_access ( GR2:A( N / 2, : ) )
        do j = 1, M
            D( N / 2, j ) = A( N / 2, j )
            isumc2 = isumc2 + C( N / 2, j )
            isuma2 = isuma2 + D( N / 2, j )
        enddo

        isumc3 = 0
        isuma3 = 0
        ki = 2
        ki1 = 3
        !dvm$ remote_access ( GR3:A1( :, M / 2 ) )
        do i = 1, N / ki - ki1
            D( i, M / 2 ) = A1( ki * i + ki1, M / 2 )
            isumc3 = isumc3 + C( ki * i + ki1, M / 2 )
            isuma3 = isuma3 + D( i, M / 2 )
        enddo

        isumc4 = 0
        isuma4 = 0
        kj = 2
        kj1 = 3
        !dvm$ remote_access ( GR3:A1( N / 2, : ) )
        do j = 1, M/kj-kj1
            D( N / 2, j ) = A1( N / 2, kj * j + kj1 )
            isumc7 = isumc7 + C( N / 2, kj * j + kj1 )
            isuma7 = isuma7 + D( N / 2, j )
        enddo

        if ( ( isumc1 .eq. isuma1 ) .and. ( isumc2 .eq. isuma2 ) .and. ( isumc3 .eq. isuma3 ) .and. &
             ( isumc4 .eq. isuma4 ) ) then
            call ansyes( tname )
        else
            call ansno( tname )
        endif

        if ( it .eq. 2 ) cycle

        !dvm$ reset GR1
        !dvm$ reset GR2
        !dvm$ reset GR3
    enddo
    deallocate( A, B, C, D, A1 )
end

subroutine prf2403
    integer, parameter ::  N = 4, M = 4, NL = 1000, NIT = 3
    integer, allocatable :: A( :, : ), B( :, : ), C( :, : ), A1( :, : )
    character * 7 :: tname ='PRF2403'

    !dvm$ distribute B( block, * )
    !dvm$ align( :, : ) with B( :, : ) :: A, A1

    !dvm$ remote_group GR1
    !dvm$ remote_group GR2
    !dvm$ remote_group GR3

    allocate ( B( N, M ), A( N, M ), C( N, M ), A1( N, M ) )
    call serial2( C, N, M, NL )

    !dvm$ parallel ( i, j ) on A( i, j )
    do i = 1, N
        do j = 1, M
            A( i, j ) = NL + i + j
            A1( i, j ) = NL + i + j
        enddo
    enddo

    do it = 1, NIT
        !dvm$ prefetch GR1
        !dvm$ prefetch GR2

        nloopi1 = NL
        nloopj1 = NL
        !dvm$ parallel ( i, j ) on B( i, j ), remote_access( GR1:A( N / 2, M / 2 ) )
        do i = 1, N
            do j = 1, M
                B( i, j ) = A( N / 2, M / 2 )
            enddo
        enddo
        !dvm$ parallel ( i, j ) on B( i, j ), reduction( min( nloopi1 ), min( nloopj1 ) )
        do i = 1, N
            do j = 1, M
                if ( B( i, j )  .ne. C( N / 2, M / 2 ) ) then
                    nloopi1 = min( nloopi1, i )
                    nloopj1 = min( nloopj1, j )
                endif
            enddo
        enddo

        nloopi2 = NL
        nloopj2 = NL
        !dvm$ parallel ( i, j ) on B( i, j ), remote_access( GR2:A1( :, M / 2 ) )
        do i = 1, N
            do j = 1, M
                B( i, j ) = A1( i, M / 2 )
            enddo
        enddo
        !dvm$ parallel ( i, j ) on B( i, j ), reduction( min( nloopi2 ), min( nloopj2 ) )
        do i = 1, N
            do j = 1, M
                if ( B( i, j ) .ne. C( i, M / 2 ) ) then
                    nloopi2 = min( nloopi2, i )
                    nloopj2 = min( nloopj2, j )
                endif
            enddo
        enddo

        nloopi3 = NL
        nloopj3 = NL
        !dvm$ parallel ( i, j ) on A( i, j ), remote_access( GR2:A1( N / 2, : ) )
        do i = 1, N
            do j = 1, M
                B( i, j ) = A1( N / 2, j )
            enddo
        enddo
        !dvm$ parallel ( i, j ) on A( i, j ), reduction( min( nloopi3 ), min( nloopj3 ) )
        do i = 1, N
            do j = 1, M
                if ( B( i, j ) .ne. C( N / 2, j ) ) then
                    nloopi3 = min( nloopi3, i )
                    nloopj3 = min( nloopj3, j )
                endif
            enddo
        enddo

        if ( ( nloopi1 .eq. NL ) .and. ( nloopj1 .eq. NL ) .and. &
             ( nloopi2 .eq. NL ) .and. ( nloopj2 .eq. NL ) .and. &
             ( nloopi3 .eq. NL ) .and. ( nloopj3 .eq. NL ) ) then
            call ansyes( tname )
        else
            call ansno( tname )
        endif

        if ( it .eq. 2 ) cycle
        !dvm$ reset GR1
        !dvm$ reset GR2
    enddo

    deallocate( A, B, C, A1 )
end

subroutine serial2( AR, N, M, NL )
    integer AR( N, M )
    integer NL
    do i = 1, N
        do j = 1, M
            AR( i, j ) = NL + i + j
        enddo
    enddo
end

subroutine ansyes( name )
    character * 7 name
    print *, name, '  -  complete'
end

subroutine ansno( name )
    character * 7 name
    print *, name, '  -  ***error'
end
