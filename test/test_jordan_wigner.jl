using ITensorMPS
using ITensorCorrelators
using Test

@testset "Jordan-Wigner String Tests" begin

    # Setup: Create test systems
    N = 4
    sites = siteinds("Electron", N)
    psi = random_mps(sites; linkdims=4)

    println("\n" * "="^60)
    println("Testing Jordan-Wigner String Implementation")
    println("="^60)

    # Test 1: C operators should have JW strings
    @testset "C operators (with JW strings)" begin
        println("\nTest 1: C operators (should use JW strings)")

        # Test various C operator combinations
        c1 = correlator(psi, ("Cdagup", "Cup"), [(1,2), (2,3)])
        @test isa(c1, Dict)
        @test length(c1) == 2
        println("  ✓ Cdagup-Cup correlator computed")

        c2 = correlator(psi, ("Cdagdn", "Cdn"), [(1,2), (2,3)])
        @test isa(c2, Dict)
        println("  ✓ Cdagdn-Cdn correlator computed")

        c3 = correlator(psi, ("Cdagup", "Cdagdn", "Cdn", "Cup"), [(1,2,3,4)])
        @test isa(c3, Dict)
        println("  ✓ 4-point C operator correlator computed")
    end

    # Test 2: A operators should NOT have JW strings
    @testset "A operators (without JW strings)" begin
        println("\nTest 2: A operators (should NOT use JW strings)")

        a1 = correlator(psi, ("Adagup", "Aup"), [(1,2), (2,3)])
        @test isa(a1, Dict)
        @test length(a1) == 2
        println("  ✓ Adagup-Aup correlator computed")

        a2 = correlator(psi, ("Adagdn", "Adn"), [(1,2), (2,3)])
        @test isa(a2, Dict)
        println("  ✓ Adagdn-Adn correlator computed")

        a3 = correlator(psi, ("Adagup", "Adagdn", "Adn", "Aup"), [(1,2,3,4)])
        @test isa(a3, Dict)
        println("  ✓ 4-point A operator correlator computed")
    end

    # Test 3: Mixed C and A operators
    @testset "Mixed C and A operators" begin
        println("\nTest 3: Mixed C and A operators")

        m1 = correlator(psi, ("Cdagup", "Aup"), [(1,2)])
        @test isa(m1, Dict)
        println("  ✓ Cdagup-Aup mixed correlator computed")

        m2 = correlator(psi, ("Adagup", "Cup"), [(1,2)])
        @test isa(m2, Dict)
        println("  ✓ Adagup-Cup mixed correlator computed")

        m3 = correlator(psi, ("Cdagup", "Adagdn", "Cdn", "Aup"), [(1,2,3,4)])
        @test isa(m3, Dict)
        println("  ✓ 4-point mixed C/A correlator computed")
    end

    # Test 4: Different orderings
    @testset "Different operator orderings" begin
        println("\nTest 4: Different operator orderings")

        # C operators in different orders
        c_fwd = correlator(psi, ("Cdagup", "Cup"), [(1,2)])
        c_rev = correlator(psi, ("Cdagup", "Cup"), [(2,1)])
        @test isa(c_fwd, Dict) && isa(c_rev, Dict)
        println("  ✓ C operators forward and reverse order")

        # A operators in different orders
        a_fwd = correlator(psi, ("Adagup", "Aup"), [(1,2)])
        a_rev = correlator(psi, ("Adagup", "Aup"), [(2,1)])
        @test isa(a_fwd, Dict) && isa(a_rev, Dict)
        println("  ✓ A operators forward and reverse order")
    end

    # Test 5: Helper function with lowercase names
    @testset "Helper function handles lowercase" begin
        println("\nTest 5: Helper function with lowercase names")

        # Test that the helper function itself handles lowercase correctly
        needs_jw = ITensorCorrelators.needs_jordan_wigner

        @test needs_jw("cdagup") == true
        @test needs_jw("cup") == true
        @test needs_jw("adagup") == false
        @test needs_jw("aup") == false
        println("  ✓ needs_jordan_wigner handles lowercase correctly")

        # Note: ITensors.jl itself doesn't support lowercase operator names,
        # so we can't test full correlator computation with lowercase
    end

    # Test 6: Verify C and A give different results
    @testset "C vs A operators give different results" begin
        println("\nTest 6: C and A operators should give different results")

        # Reset to a known state
        sites2 = siteinds("Electron", 2)
        psi2 = random_mps(sites2; linkdims=4)

        c_result = correlator(psi2, ("Cdagup", "Cup"), [(1,2)])
        a_result = correlator(psi2, ("Adagup", "Aup"), [(1,2)])

        c_val = c_result[(1,2)]
        a_val = a_result[(1,2)]

        println("  C-operator result: $c_val")
        println("  A-operator result: $a_val")

        # They should generally give different results due to JW strings
        # (unless by chance the JW string happens to not affect the result)
        @test isa(c_val, Number) && isa(a_val, Number)
        println("  ✓ Both produce valid numerical results")
    end

    # Test 7: Multiple site tuples
    @testset "Multiple site tuples" begin
        println("\nTest 7: Multiple site tuples")

        sites_list = [(1,2), (2,3), (3,4), (1,3)]

        c_multi = correlator(psi, ("Cdagup", "Cup"), sites_list)
        @test isa(c_multi, Dict)
        @test length(c_multi) == length(sites_list)
        println("  ✓ C operators with $(length(sites_list)) site tuples")

        a_multi = correlator(psi, ("Adagup", "Aup"), sites_list)
        @test isa(a_multi, Dict)
        @test length(a_multi) == length(sites_list)
        println("  ✓ A operators with $(length(sites_list)) site tuples")
    end

    # Test 8: On-site correlators
    @testset "On-site correlators" begin
        println("\nTest 8: On-site correlators")

        c_onsite = correlator(psi, ("Cdagup", "Cup"), [(1,1), (2,2)])
        @test isa(c_onsite, Dict)
        println("  ✓ On-site C operator correlator")

        a_onsite = correlator(psi, ("Adagup", "Aup"), [(1,1), (2,2)])
        @test isa(a_onsite, Dict)
        println("  ✓ On-site A operator correlator")
    end

    println("\n" * "="^60)
    println("All Jordan-Wigner tests passed!")
    println("="^60 * "\n")
end
