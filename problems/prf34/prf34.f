program prf34
    !TESTING OF THE PREFETCH DIRECTIVE.

    print *, '===START OF PRF34========================'

    call prf3401
    call prf3402
    call prf3403

    print *, '===END OF PRF34=========================='
end

subroutine prf3401
    integer, parameter ::  N = 16, M = 8, K = 8, NL = 1000, NIT = 3
    integer, allocatable :: A( :, :, : ), B( :, :, : ), C( :, :, : )
    character * 7 :: tname = 'PRF3401'

    !dvm$ distribute B( block, block, * )
    !dvm$ align( :, :, : ) with B( :, :, : ) :: A

    !dvm$ remote_group GR1
    !dvm$ remote_group GR2
    !dvm$ remote_group GR3

    allocate ( B( N, M, K ), A( N, M, K ), C( N, M, K ) )
    call serial3( C, N, M, K, NL )

    !dvm$ parallel ( i, j, ii ) on A( i, j, ii )
    do i = 1, N
        do j = 1, M
            do ii = 1, K
                A( i, j, ii ) = NL + i + j + ii
            enddo
        enddo
    enddo

    do it = 1, NIT
        !dvm$ prefetch GR1
        !dvm$ prefetch GR2
        !dvm$ prefetch GR3

        !dvm$ remote_access ( GR1:A( N / 2, M / 2, K / 2 ) )
        ib1 = A( N / 2, M / 2, K / 2 )

        !dvm$ remote_access ( GR1:A( N / 2, M, K ) )
        ib2 = A( N / 2, M, K )

        !dvm$ remote_access ( GR2:A( N, M / 2, K ) )
        ib3 = A( N, M / 2, K )

        !dvm$ remote_access ( GR2:A( N, M, K / 2 ) )
        ib4 = A( N, M, K / 2 )

        !dvm$ remote_access ( GR3:A( N / 2, M, 1 ) )
        ib5 = A( N / 2, M, 1 )

        if ( ( ib1 .eq. C( N / 2, M / 2, K / 2 ) ) .and. ( ib2 .eq. C( N / 2, M, K ) ) .and. &
             ( ib3 .eq. C( N, M / 2, K ) ) .and. ( ib4 .eq. C( N, M, K / 2 ) ) .and. &
             ( ib5 .eq. C( N / 2, M, 1 ) ) ) then
            call ansyes( tname )
        else
            call ansno( tname )
        endif

        if ( it .eq. 2 ) cycle
        !dvm$ reset GR1
        !dvm$ reset GR2
        !dvm$ reset GR3
    enddo
    deallocate( A, B, C )
end

subroutine prf3402
    integer, parameter :: N = 16, M = 8, K = 8, NL = 1000, NIT = 3
    integer, allocatable :: A( :, :, : ), B( :, :, : ), C( :, :, : ), D( :, :, : )
    character * 7 :: tname = 'PRF3402'

    !dvm$ distribute A( block, *, block )
    !dvm$ align( :, :, : ) with A( :, :, : ) :: B

    !dvm$ remote_group GR1
    !dvm$ remote_group GR2
    !dvm$ remote_group GR3

    allocate ( A( N, M, K ), B( N, M, K ), C( N, M, K ), D( N, M, K ) )
    call serial3( C, N, M, K, NL )

    !dvm$ parallel ( i, j, ii ) on A( i, j, ii )
    do i = 1, N
        do j = 1, M
            do ii = 1, K
                A( i, j, ii ) = NL + i + j + ii
            enddo
        enddo
    enddo

    do it = 1, NIT
        !dvm$ prefetch GR1
        !dvm$ prefetch GR2
        !dvm$ prefetch GR3

        isumc1 = 0
        isuma1 = 0
        !dvm$ remote_access ( GR1:A( N / 2, :, : ) )
        do j = 1, M
            do ii = 1, K
                D( N / 2, j, ii ) = A( N / 2, j, ii )
                isumc1 = isumc1 + C( N / 2, j, ii )
                isuma1 = isuma1 + D( N / 2, j, ii )
            enddo
        enddo

        isumc2 = 0
        isuma2 = 0
        !dvm$ remote_access ( GR1:A( :, M / 2, : ) )
        do i = 1, N
            do ii = 1, K
                D( i, M / 2, ii ) = A( i, M / 2, ii )
                isumc2 = isumc2 + C( i, M / 2, ii )
                isuma2 = isuma2 + D( i, M / 2, ii )
            enddo
        enddo

        isumc3 = 0
        isuma3 = 0
        !dvm$ remote_access ( GR2:A( :, :, K / 2 ) )
        do i = 1, N
            do j = 1, M
                D( i, j, K / 2 ) = A( i, j, K / 2 )
                isumc3 = isumc3 + C( i, j, K / 2 )
                isuma3 = isuma3 + D( i, j, K / 2 )
            enddo
        enddo

        isumc4 = 0
        isuma4 = 0
        kj = 2
        kj1 = 3
        kii = 2
        kii1 = 3
        !dvm$ remote_access ( GR3:A( N / 2, :, : ) )
        do j = 1, M / kj-kj1
            do ii = 1, K / kii-kii1
                D( N / 2, j, ii ) = A( N / 2, kj * j + kj1, kii * ii + kii1 )
                isumc4 = isumc4 + C( N / 2, kj * j + kj1, kii * ii + kii1 )
                isuma4 = isuma4 + D( N / 2, j, ii )
            enddo
        enddo

        if ( ( isumc1 .eq. isuma1 ) .and. ( isumc2 .eq. isuma2 ) .and. &
             ( isumc3 .eq. isuma3 ) .and. ( isumc4 .eq. isuma4 ) ) then
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

subroutine prf3403
    integer, parameter ::  N = 16, M = 8, K = 8, NL = 1000, NIT = 3
    integer, allocatable :: A( :, :, : ), B( :, :, : ), C( :, :, : ), A1( :, :, : )
    character * 7 :: tname = 'PRF3403'

    !dvm$ distribute A( *, block, block )
    !dvm$ align( :, :, : ) with A( :, :, : ) :: B, A1

    !dvm$ remote_group GR1
    !dvm$ remote_group GR2
    !dvm$ remote_group GR3

    allocate ( A( N, M, K ), B( N, M, K ), C( N, M, K ), A1( N, M, K ) )
    call serial3( C, N, M, K, NL )

    !dvm$ parallel ( i, j, ii ) on A( i, j, ii )
    do i = 1, N
        do j = 1, M
            do ii = 1, K
                A( i, j, ii ) = NL + i + j + ii
                A1( i, j, ii ) = NL + i + j + ii
            enddo
        enddo
    enddo

    do it = 1, NIT
        !dvm$ prefetch GR1
        !dvm$ prefetch GR2

        nloopi1 = NL
        nloopj1 = NL
        nloopii1 = NL
        !dvm$ parallel ( i, j, ii ) on B( i, j, ii ), remote_access( GR1:A( N / 2, M / 2, K / 2 ) )
        do i = 1, N
            do j = 1, M
                do ii = 1, K
                    B( i, j, ii ) = A( N / 2, M / 2, K / 2 )
                enddo
            enddo
        enddo
        !dvm$ parallel ( i, j, ii ) on B( i, j, ii ), reduction( min( nloopi1 ), min( nloopj1 ), min( nloopii1 ) )
        do i = 1, N
            do j = 1, M
                do ii = 1, K
                    if ( B( i, j, ii ) .ne. C( N / 2, M / 2, K / 2 ) ) then
                        nloopi1 = min( nloopi1, i )
                        nloopj1 = min( nloopj1, j )
                        nloopii1 = min( nloopii1, ii )
                    endif
                enddo
            enddo
        enddo

        nloopi2 = NL
        nloopj2 = NL
        nloopii2 = NL
        !dvm$ parallel ( i, j, ii ) on A( i, j, ii ), remote_access( GR1:A( N / 2, :, : ) )
        do i = 1, N
            do j = 1, M
                do ii = 1, K
                    B( i, j, ii ) = A( N / 2, j, ii )
                enddo
            enddo
        enddo
        !dvm$ parallel ( i, j, ii ) on B( i, j, ii ), reduction( min( nloopi2 ), min( nloopj2 ), min( nloopii2 ) )
        do i = 1, N
            do j = 1, M
                do ii = 1, K
                    if ( B( i, j, ii ) .ne. C( N / 2, j, ii ) ) then
                        nloopi2 = min( nloopi2, i )
                        nloopj2 = min( nloopj2, j )
                        nloopii2 = min( nloopii2, ii )
                    endif
                enddo
            enddo
        enddo

        nloopi3 = NL
        nloopj3 = NL
        nloopii3 = NL
        !dvm$ parallel ( i, j, ii ) on B( i, j, ii ), remote_access( GR2:A1( :, M / 2, : ) )
        do i = 1, N
            do j = 1, M
                do ii = 1, K
                    B( i, j, ii ) = A1( i, M / 2, ii )
                enddo
            enddo
        enddo
        !dvm$ parallel ( i, j, ii ) on B( i, j, ii ), reduction( min( nloopi3 ), min( nloopj3 ), min( nloopii3 ) )
        do i = 1, N
            do j = 1, M
                do ii = 1, K
                    if ( B( i, j, ii ) .ne. C( i, M / 2, ii ) ) then
                        nloopi3 = min( nloopi3, i )
                        nloopj3 = min( nloopj3, j )
                        nloopii3 = min( nloopii3, ii )
                    endif
                enddo
            enddo
        enddo

        nloopi4 = NL
        nloopj4 = NL
        nloopii4 = NL
        !dvm$ parallel ( i, j, ii ) on A( i, j, ii ), remote_access( GR2:A1( :, :, K / 2 ) )
        do i = 1, N
            do j = 1, M
                do ii = 1, K
                    B( i, j, ii ) = A1( i, j, K / 2 )
                enddo
            enddo
        enddo
        !dvm$ parallel ( i, j, ii ) on A( i, j, ii ), reduction( min( nloopi4 ), min( nloopj4 ), min( nloopii4 ) )
        do i = 1, N
            do j = 1, M
                do ii = 1, K
                    if ( B( i, j, ii ) .ne. C( i, j, K / 2 ) ) then
                        nloopi4 = min( nloopi4, i )
                        nloopj4 = min( nloopj4, j )
                        nloopii4 = min( nloopii4, ii )
                    endif
                enddo
            enddo
        enddo

        if ( ( nloopi1 .eq. NL ) .and. ( nloopi2 .eq. NL ) .and. &
             ( nloopi3 .eq. NL ) .and. ( nloopi4 .eq. NL ) ) then
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

subroutine serial3( AR, N, M, K, NL )
    integer AR( N, M, K )
    integer NL
    do i = 1, N
        do j = 1, M
            do ii = 1, K
            AR( i, j, ii ) = NL + i + j + ii
            enddo
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
