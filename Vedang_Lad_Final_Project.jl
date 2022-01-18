### A Pluto.jl notebook ###
# v0.14.4

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ 3d6fb095-69d6-4345-b899-d50d07eb8955
using LinearAlgebra, Plots, PlutoUI

# ╔═╡ e2cd8d91-1281-44c4-a953-c45114174521
TableOfContents()

# ╔═╡ 013dd4ed-9eda-4d80-b1b1-2cf9cb2ffc68
md"""
## Final Project: An Investigation of Hyperbolic PDEs and FTCS/Lax Methods 
By Vedang Lad 

"""

# ╔═╡ 70f335de-b771-4c00-8785-56b2be24b6a1
md"""
### Part 1: FTCS for Hyperbolic PDEs
Here we will now tackle the challenge of partial differential equations, or PDEs. We will start with some some simple rudimentary methods, look at the stability, and try to tweak our method to a better solution then perform some error analysis.

In reality, with the importance of PDEs in finance, physics, and engineering there is a plethora of complex literature on the solving of PDEs. In this notebook I hope to capture a small glimpse of this information, when in reality one could create a course dedicated to just PDEs. Here we will focus on Hyperbolic PDEs, specifically the Advection Equation
"""

# ╔═╡ d61fb5c6-5e16-43a7-bea1-01d65a9c9c9f
md"""
Starting with the simplest PDE, we will explore the Flux-Conservative Problem. This is also known as the Advection Equation. It takes the form:

$c\frac{∂u}{∂x}=\frac{∂u}{∂t}$

The flux, defined as the number of field lines that cross the plane of the surface, means that this equation implies the conservation of particles inside a volume. Specifically, the number of flowing "particles" inside of some finite volume is the same as the flux of the current associated to the movement of these particles.

"""

# ╔═╡ e0f8ed89-9ee2-44ca-89c0-9fa7760cbf8b
md"""
We will start with an explicit scheme, using the Forward Time Centered Space representation, or (FTCS)
"""

# ╔═╡ c5fce78f-71c4-4be6-beeb-3c6788f6d088
md"""
Using the (forward) euler approximation, we see that 


$\frac{∂u}{∂t}=\frac{u_{n+1,j}-u_{n,j}}{\Delta t} + O(\Delta t)$

We can also use the "information" we have at time step $j$ to have a second order represenatation of the $\frac{du}{dx}$. Let us derive this:


Now using some Taylor Expansions:

$u(x_j+\Delta x, t^n) = u(x_j,t^n)+\frac{∂u}{∂x}(x_j,t^n)\Delta x+\frac{1}{2}\frac{∂^2u}{∂x^2}(x_j,t^n)\Delta x^2+O(\Delta x^3)$

$u(x_j-\Delta x, t^n) = u(x_j,t^n)-\frac{∂u}{∂x}(x_j,t^n)\Delta x+\frac{1}{2}\frac{∂^2u}{∂x^2}(x_j,t^n)\Delta x^2+O(\Delta x^3)$

Substracting the two and dividing by $2 \Delta x$ we can eliminate the first order terms to get that 

$\frac{∂u}{∂x}=\frac{u_{n,j+1}-u_{n,j-1}}{2\Delta x}+ O(\Delta x^2)$
This is the central difference scheme!


Combinining these two results with differential yields:

$-c* \frac{u_{n,j+1}-u_{n,j-1}}{2\Delta x}= \frac{u_{n+1,j}-u_{n,j}}{\Delta t}$


$u_{n+1,j} = u_{n,j} -\frac{c\Delta t}{2 \Delta x}(u_{n,j+1} -u_{n,j-1})$

Also note that from our derivations above we see that this method is first order in time and second order in space!

Let us implement this method and explore its stability by looking at the evolution of a square wave!

"""

# ╔═╡ 9a7ba6fb-d2d6-4e96-a4b4-3fd4337bd215
md"""
We will use this initialization again so we will just make it once to be reused again!
"""

# ╔═╡ 610c2ee9-23f3-459e-8e7b-c25f88c4961f
function initialization(maxL, maxT, c, tsteps, xsteps, size)
#maxL is the maximum length
#maxT is the maximum time
#c is the velocity of the advection (i.e in wave eq c=speed of light for EM wave)

#tsteps is the number of time steps
#xsteps is the number of space steps
#size the wavefront

	dt=maxT/tsteps
	dx=maxL/xsteps
	factor=c*dt/(2.0*dx)
	u=zeros(xsteps+1,tsteps+1)#A U matrix to store our solution
	x=[0.0 for j in 1:xsteps+1]
	t=[0.0 for j in 1:tsteps+1]
	
	#Set the initial value of our solution u, determining the amplitude
	for i in 1:xsteps+1
		if i < size
			u[i,1]=1.0 #When less than size the square wave is 1
		else 
			u[i,1]=0.0 
		end
		x[i] = (i-1)*dx
	end
	
	#Determine the value of the amplitude at the boundary
	for k in 1:tsteps+1
		u[1,k] = 1.0 
		t[k] = (k-1)*dt
		u[xsteps+1,k]=0.0 
	end
	
	return dt, dx, factor, u,x,t
end

# ╔═╡ e2572c66-0256-46f4-90f4-ef31b57e95b4
md"""
Note that this algorithm is quite simple - yet effective. Or is it even effective?
"""

# ╔═╡ d75cdfa2-65a5-44b1-9595-6ce4c9555381
function FTCS(maxL, maxT, c, tsteps, xsteps, size)
	#The FTCS method
	#let us use our initialization above
	dt, dx, factor, u,x,t = initialization(maxL, maxT, c, tsteps, xsteps, size)
	
	for k in 1:tsteps #time
		for i in 2:xsteps#space
			u[i,k+1] = u[i,k]-factor*(u[i+1,k]-u[i-1,k]) #Our method!
		end
	end
	
	return x,u,t	
end
	

# ╔═╡ f6db5bfa-1388-4704-bae7-81690c989594
md"""
Let us now see what all these numbers mean!
"""

# ╔═╡ fe04a936-e372-4adb-9724-c8da6df4d78e
md"""
Toggle c, which our velocity. By picking a large velocity we can explore later time evolution (since the square wave will move faster)
"""

# ╔═╡ 7bd3885d-a749-4047-9331-6aed47f5675c
@bind c Slider(0:1:100,show_value=true, default=10)

# ╔═╡ 574ac318-8d4d-4863-bd88-9d8dcb67eeb1
x,u,t=FTCS(1.0, 1.0, c, 3000, 30, 15)

# ╔═╡ 66162018-e21d-4e8f-b3f7-3aed8de06676
begin
	plot(x,u[:,1],title="Square Wave Propagation with Advection Equation and FTCS",xlabel="x",ylabel="u",label="t=1")
	plot!(x,u[:,10],label="t=10")
	plot!(x,u[:,30],label="t=30")
	plot!(x,u[:,100],label="t=100")
end

# ╔═╡ f697c6ba-5076-447a-ac06-28d6d0c3efe2
md"""
Use the slider to change the velocity of the wave so we can explore later time steps. Clearly this method is HIGHLY unstable - as the solution explodes and becomes highly ocillatory regardless of our choice of $\Delta x$ and $\Delta t$ (Note here we are changing c, we can also change $\Delta x$ and $\Delta t$ to achieve the same results - they have the same affect of the equation of motion)

One can use a Fourier approach to show this explosion mathematically. This is is called the Von-Neamann Stability analysis.

The result involves decompsing into Fourier modes 

$u_{n,j} = \sum u_m (t_j)e^{ik_m(n\Delta x)}$

where we define $k_m$ to be the wavenumber of the $m$th mode
This is simplified to 

$u_{n,j} = E(k_m)e^{ik_m(n\Delta x)}$

since the Fourier modes are uncoupled so we can consider each Fourier mode individually. Note that this Fourier Decomposition does not take into account the boundary conditions

In the case that $|E(k_m)|>1$, then the method is unstable.

For the method we see above, plugging $u_{n,j} = E(k_m)e^{ik_m(n\Delta x)}$ into the Advection Equation gives

$E(k_m) = 1-i\frac{c\Delta t}{\Delta x}sin(k \Delta x)$

Taking the $|E(k_m)|$ (Complex magnitude) we see that regardless of our choice of $\Delta x$ or $\Delta t$, our solution will become unstable. Yikes!

Now let us look for a fix!

"""

# ╔═╡ 98599ee3-4785-4015-96a9-38b892fdd1fb
x1,u1,t1=FTCS(1.0, 1.0, 15, 3000, 30, 15)

# ╔═╡ 9b6c6ba6-e673-469a-bf4e-dff5a853d68b
md"""
Here is a cool plot showing the instability!
"""

# ╔═╡ 56510602-a7b2-430d-8a67-ad557746d2ac
begin
	gr()
	@gif for i ∈ 1:500
		plot(x1,u1[:,i],title="Time Evolution of Advection Solution using FTCS until t=500",xlabel="x",ylabel="u",ylims=(-5,5), label="t=$i")
	end 
end


# ╔═╡ a3874dd6-cfe9-4abb-990a-d2550999a68e
md"""
The stability problem we see arises due to the fact that we are using finite computer arthmetic to solve our equations which introduces rounding errors. We can see above that these rounding errors are magnified at every iteration. Instead of decaying away, these errors get larger and larger leading to the unstable method above.
"""

# ╔═╡ 3a47e817-4f80-4750-82ac-e98d2f4008d2
md"""
### Part 2: Curing the Instability: Lax Method

A method that does not work is not very interesting. Let us try and fix this instability

To address this instability we have to alter the approximations that we use in either our spatial or time derivative approximations.

We use a new approximation for the time derivative by changing $u_{n,j}$ to its average $\frac{u_{n,j+1}+u_{n,j-1}}{2}$

Thus our old forward euler approximation of 

$\frac{∂u}{∂t}=\frac{u_{n+1,j}-u_{n,j}}{\Delta t} + O(\Delta t)$
becomes

$\frac{∂u}{∂t}=\frac{u_{n+1,j}-\frac{u_{n,j+1}+u_{n,j-1}}{2}}{\Delta t} + O(\Delta t)$

Thus our new approximation yields

$u_{n+1,j} = \frac{u_{n,j+1}+u_{n,j-1}}{2} -\frac{c\Delta t}{2 \Delta x}(u_{n,j+1} -u_{n,j-1})$

This is called the Lax-Friedrichs method, which is a numerical method for the solution of hyperbolic PDEs which is based in just finite differences

Notice that this method is also explicit! We we apply the same von Neumann stability analysis above where $u_{n,j} = E(k_m)e^{ik_m(n\Delta x)}$ 

for the amplification factor, after some quick calculus, we get 

$E(k_m) = cos(k_m \Delta x)-i\frac{c\Delta t}{\Delta x}sin(k_m \Delta x)$

This means that our method will remain stable if:

$\frac{|c|\Delta t}{\Delta x}\leq 1$

This condition, which also arises when solving PDEs is known as the Courant–Friedrichs–Lewy condition or CFL condition for short

Let us see what this looks like!


"""


# ╔═╡ ee882691-14a9-4085-8a8a-01a60275915b
md"""
Our entire method will stay the same as before except now...

$u[i,k+1] = 0.5*(u[i+1,k]+u[i-1,k])-b*(u[i+1,k]-u[i-1,k])$

This is changed below in the "third" part of our method

"""

# ╔═╡ e98a6e02-3af8-4366-80a0-0578c91f3353
#maxL is the maximum length
#maxT is the maximum time
#c is the velocity of the advection (i.e in wave eq c=speed of light for EM wave)

#tsteps is the number of time steps
#xsteps is the number of space steps
#size the wavefront

function LAX(maxL, maxT, c, tsteps, xsteps, size)
	
	dt, dx, factor, u,x,t = initialization(maxL, maxT, c, tsteps, xsteps, size)
	
	#The LAX method
	for k in 1:tsteps #time
		for i in 2:xsteps#space
			#The line that we are changing!
			u[i,k+1] = 0.5*(u[i+1,k]+u[i-1,k])-factor*(u[i+1,k]-u[i-1,k])
		end
	end
	
	return x,u,t	
end
	

# ╔═╡ a619d0ed-ece3-40cb-b31b-d1b8e1712eec
md"""
Now we must make sure that we satisfy our CFL condition.
That 

$\frac{|c|\Delta t}{\Delta x}\leq 1$

$\frac{c*maxT*xsteps}{tsteps*maxL} \leq 1$

As seen in the code, we have the lines 

$dt=maxT/tsteps$
$dx=maxL/xsteps$

Since everything is $=1.0$ except $xsteps$ and $tsteps$ is ≠ 1 we need 

$\frac{xsteps}{tsteps} \leq 1$


"""

# ╔═╡ 3ffe4d1a-15ce-4224-bccb-69dcb461cfbc
md"""
Let us propagate a wave where  
 
$maxL = 1.0$

$maxT = 1.0$

$c = 1.0$

$tstep = 350$

$xsteps = 300$

$size = 50$
"""

# ╔═╡ 5539e8e1-73df-4179-82d1-857c5c78a11d
x2,u2,t2=LAX(1.0, 1.0, 1.0, 350, 300,50)

# ╔═╡ 6e8c1180-ae88-44f0-b562-c79812ad3e06
begin
	gr()
	plot(x2,u2[:,1],title="Square Wave Propagation with Advection Equation and LAX",xlabel="x",ylabel="u",label="t=1")
	plot!(x2,u2[:,10],label="t=10")
	plot!(x2,u2[:,30],label="t=30")
	plot!(x2,u2[:,100],label="t=100")
end

# ╔═╡ 9c7e0eae-6697-421d-948e-853ce8cec511
md"""
This looks like a square wave propagating. We can also see that with our choice of dx and dt it is not a "perfect" square wave. We will investigate this later!
"""

# ╔═╡ 40bdbb79-5db0-40b1-bfa2-2d10e9c7c6a1
begin
	plotly()
	plot(x2,t2,u2,st=:surface, label="x", ylabel="t",zlabel="u", camera=(110,10),title="3D Plot of Square Wave Propagation using LAX")
end

# ╔═╡ b2395666-176c-4026-87e7-ac14bef0c234
begin
	gr()
	@gif for i ∈ 1:300
		plot(x2,u2[:,i],title="Time Evolution of Advection Solution using LAX until t=300",xlabel="x",ylabel="u",ylims=(-5,5), label="t=$i")
	end 
end



# ╔═╡ f7321227-4305-4f9a-9c12-abffea95b22a
md"""
Now that we have a method that works, we must perform some time of numerical analysis - a way to quantify our error.

With the plots we can clearly see that the square form does not remain square in the propagation. 

We can fix a moment in time and look see what happens when we change the spatial steps.


"""

# ╔═╡ 4907a8b6-a136-41df-9f3e-c538d50a3185
md"""
The true solution of the propagation of the square wave can be seen when set the number of time steps = number of space steps: 

This makes sense since we are saturating the amplication factor of the fourier analysis

$\frac{|c|\Delta t}{\Delta x}\leq 1$

Where 

$\Delta t = \Delta x$ 
and 

$c=1$

Note that at the this "perfect" solution occurs because of the binary nature of our propagation: the values are 1 or 0. In the case where we propagate, say a Gaussian, setting $\Delta t = \Delta x$ will not necessarily produce the true solution since there are likely to be truncation error. 

(We can only do this by taking advantage of the square wave!)

Here is some proof that $\Delta t = \Delta x$ results in the true square wave propagation:


"""

# ╔═╡ 4e4ce48b-ffe0-4ed8-be30-a3f1f2e3a207
x_true,u_true,t_true=LAX(1.0, 1.0, 1.0, 350, 350,50)

# ╔═╡ 275db70e-b78d-4f72-a0a6-ee3356ac00b7
heatmap(u_true,title="U Solution Matrix of True Solution")

# ╔═╡ 4ae051ce-7431-414a-8e7e-fb6c2c02ae11
md"""
Above we can see that the values are only 1 (Yellow) or 0 (Black)
"""

# ╔═╡ a4895d9d-2b7e-4662-aabf-acdfd8527622
begin
	gr()
	@gif for i ∈ 1:300
		plot(x_true,u_true[:,i],title="Time Evolution of True Solution of Square Wave",xlabel="x",ylabel="u",ylims=(-5,5), label="t=$i")
	end 
end



# ╔═╡ 6a58531a-93b9-478c-996e-314341886475
md"""
Let us overlay both of these plots on top of each other to see what is going on:
"""

# ╔═╡ 58d9c9b7-533a-4418-9ab7-501da86450d9
begin
	gr()
	@gif for i ∈ 1:300
		plot(x_true,u_true[:,i],title="Time Evolution of True Solution and Smaller Step Sol",xlabel="x",ylabel="u",ylims=(-5,5), label="t=$i")
		plot!(x2,u2[:,i],label="tstep=350, xstep=300" )
		plot!()
	end 
end




# ╔═╡ 571459f0-55d7-41a5-9fbd-416ca8260f29
md"""
Interesting - it looks like as we propagte through time the wave seems to "spread out" a little bit. Let us investigate this dispersive effect further
"""

# ╔═╡ 90515297-fed3-404b-931b-36c179a7d0bc
md"""
Let us now create a function so that we can analyze the error in this method!
"""

# ╔═╡ 79533303-d400-403f-920e-9f75abd97330
md"""
### Part 3: Error Analysis
"""

# ╔═╡ bc6f5299-7b9c-4a21-91e5-ce0105f6d318
md"""
To see the error in this method, we have to alter the ratio of $\frac{\Delta x}{\Delta t}$

The implemeneted funtion below asks for a timestep size and alters the xsteps and shows what happens when we keep the ratio in the stable regime (See the CFL check above).
"""

# ╔═╡ 9312fad0-c2ce-437c-b1c2-8a68a6f77a83
function error_at_timestep(timestep,maxL, maxT, c, tsteps, xsteps, size)
	#timestep is the step in time we want to analyze
	#assume tsteps=xsteps in the input
	error=[]
	
	#perfect square wave propagation
	utrue=u_result=LAX(maxL, maxT, c, xsteps, tsteps, size)[2]
	
	
	for i in 1:100
		#we will decrease our time steps by holding the xsteps constant and increasing the tsteps
		
		
		#With each iteration we make delta t larger, thus our error should go up
		u_result=LAX(maxL, maxT, c, xsteps-i, tsteps, size)[2]
		
		#We take the maximum of the the abs. on the interval
		push!(error, maximum(abs.(u_true[:,timestep]-u_result[:,timestep])))
		
		
	end
	
	xs=[maxT/(tsteps-i) for i in 1:100]

	return error,xs
	
end

# ╔═╡ 79438e59-c394-4011-b921-d8406becb7df
begin
	error,xs = error_at_timestep(2,1.0, 1.0, 1.0, 350, 350,50)
	plot(xs,error,xlabel="ΔX", ylabel="Difference in Solution U",title="Error of Solution at timestep 2")
end



# ╔═╡ f8f49dc1-5543-4956-9ce9-c39d399cc378
begin
	error2,xs2 = error_at_timestep(100,1.0, 1.0, 1.0, 350, 350,50)
	plot(xs2,error2,xlabel="ΔX", ylabel="Difference in Solution U",title="Error of Solution at timestep 100")
end




# ╔═╡ 431c651e-13e8-463b-9b1a-57e2b6337d1f
md"""
Well this is interesting. We seem to be seeing this increase in error as the wave is propagating to the right. We can also see this in the propagation of the two wave in the gif above. As the wave moves to the right it seem to shape more and more. Let us see the full animation of this error:
"""

# ╔═╡ e930562c-0041-45db-adcb-b038f5cd13d2
begin
	gr()
	@gif for i ∈ 1:50
		error3,xs3 = error_at_timestep(i,1.0, 1.0, 1.0, 350, 350,50)
		plot(xs3,error3,xlabel="ΔX", ylabel="Difference in Solution U",title="Error of Solution at timestep $i", label="timestep=$i")
	end 
end

# ╔═╡ aa272063-dda3-415d-a5a2-992bd468a690
md"""
The error of the LAX method is infact a well known phenomenon. This is called numerical dissapation, or viscosity in the solution. This arises from the average substitution that we make going from the FTCS method to the LAX method. 

To see this we need to analyze our substitution further:

We rewrite 

$u_{n+1,j} = \frac{u_{n,j+1}+u_{n,j-1}}{2} -\frac{c\Delta t}{2 \Delta x}(u_{n,j+1} -u_{n,j-1})$

so that we can see a "correction term"

$\frac{u_{n+1,j}-u_{n,j}}{\Delta t} = -c* (\frac{u_{n,j+1}-u_{n,j-1}}{2\Delta x}) + \frac{1}{2}(\frac{u_{n,j+1}-2u_{n,j}+u_{n,j-1}}{\Delta t})$

However this is exactly the finite difference representation of the equation

$\frac{∂u}{∂t}+c\frac{du}{dx} = \frac{1}{2}(\frac{\Delta x^2}{\Delta t})\frac{∂^2u}{∂x^2}$


To find out what

$\frac{∂^2u}{∂x^2}$ is we can reuse our Taylor Expansions:


$u(x_j+\Delta x, t^n) = u(x_j,t^n)+\frac{∂u}{∂x}(x_j,t^n)\Delta x+\frac{1}{2}\frac{∂^2u}{∂x^2}(x_j,t^n)\Delta x^2+O(\Delta x^3)$

$u(x_j-\Delta x, t^n) = u(x_j,t^n)-\frac{∂u}{∂x}(x_j,t^n)\Delta x+\frac{1}{2}\frac{∂^2u}{∂x^2}(x_j,t^n)\Delta x^2+O(\Delta x^3)$

Summing this and solving for $\frac{d^2u}{dx^2}$ we get that 

$\frac{∂^2u}{∂x^2} =(\frac{u_{n,j+1}-2u_{n,j}+u_{n,j-1}}{\Delta x^2})+O(\Delta x^2)$

Putting it all together, we have 

$\frac{∂u}{∂t}+c\frac{∂u}{∂x} = \frac{1}{2}(\frac{\Delta x^2}{\Delta t})\frac{∂^2u}{∂x^2}$

where 

$\frac{∂^2u}{∂x^2} =(\frac{u_{n,j+1}-2u_{n,j}+u_{n,j-1}}{\Delta x^2})+O(\Delta x^2)$

Now earlier, we showed that the expression for $\Delta x^2$ is $O(\Delta x^2)$ simply by the Euler forward method and that $\Delta t$ is $O(\Delta t)$ (using the Taylor Expansions with finite difference).

Thus the $\frac{\Delta x^2}{\Delta t}$ term cancels and is left to be overall $O(\Delta x)$. This multiplied by the $\frac{∂^2u}{∂x^2}$, which is $O(\Delta x^2)$ makes the right hand side $O(\Delta x^3)$


This is the error that we are seeing!

For more information about this see reference [5]!













"""

# ╔═╡ f6707b0a-940d-4954-85af-14cc9a305165
md"""
The following plots are just to show that the general shape matches that of a relation that is $O(\Delta x^3)$. Ignore the axis as they have not be scaled appropriately using initial conditions; these plots characterize the error we see above. At an early timestep, there is little dispersion. The wave has not been in such a "artificial viscosity" for a long amount of time
"""

# ╔═╡ c2da3bbf-05c0-42b8-81ce-9860d5cb5e87
plot([1/(350.0-x)^3 for x in 1:100],title="O(Δx^3 Error at Small Time step)",leg=false)

# ╔═╡ 90c249fd-1255-42f0-81c9-3dcca77b1438
md"""
Note that ΔX is equal to 1/(350.0-x)
"""

# ╔═╡ 4f1b8e9a-8529-49ac-aa8d-d385903b7477
md"""
Again ignoring the axis, we can now shift as far as 300 away from the optimal value. This is of a similar shape as the error we saw above!
"""

# ╔═╡ 0d9774f3-7baa-4e82-a6ec-340b41095c5f
plot([1/(350-x)^3 for x in 1:300],title="O(Δx^3 Error at Large Time step)",leg=false)

# ╔═╡ 3d2b47a9-dc84-40c6-b0c8-e7fe533d8856
md"""
Note that ΔX is equal to 1/(350.0-x)
"""

# ╔═╡ 67eb179e-c78d-4c3e-9016-d4bb74ab47c1
md"""
A more interesting discussion of the error in this method takes places in the paper [1], where a two-step scheme is proposed in order to slow dispersion and dissipation, this paper published by NASA investigates yet another update on this method! This method is called the Lax–Wendroff Method which has fewer effects of the this artifical viscosity, which is achieved by keeping more terms in the Taylor expansions we showed above.

While [1] is more thorough in the numerical analysis of the the method, [4] shows rigorous mathematical proofs. The error analysis of this method is better seen in [7]. Here we learn, intuitively, that keeping more terms in Taylor expansions gives us a higher degree of accuracy in space and time.

It is important to note that neither of these methods are the most practical. They are computationally expensive and have instabilities that have to be delt with.

As we discussed in class, oftentimes we find ourselves converting problems to a linear problem. The same can be done here and we can simply solve the propagation of the PDE using LU decomposition

This is known as the Crank-Nicholson method! This method is stable in all regime which is what makes it especially useful!

To conclude, while this is certainly not the most optimal method, it is certainly a great start to learn from!


"""

# ╔═╡ 9bbf668b-edbb-48bd-9ad4-d68829c463ce
md"""
(Some of the references dont work from the MD text output and have to be copied from the code)
"""

# ╔═╡ 15efbb34-ed94-4cf9-a660-00638436a246
md"""
References:

[1] https://journals.ametsoc.org/view/journals/mwre/113/6/1520-0493_1985_113_1050_atssft_2_0_co_2.xml?tab_body=pdf


[2] https://www.esaim-m2an.org/articles/m2an/pdf/2004/03/m2an0383.pdf

[3] Numerical Analysis of Partial Differential Equations by S.H Lui
https://onlinelibrary.wiley.com/doi/pdf/10.1002/9781118111130

[4] https://warwick.ac.uk/fac/cross_fac/complexity/study/msc_and_phd/co906/co906online/lecturenotes_2009/chap4.pdf

[5] https://itp.uni-frankfurt.de/~rezzolla/lecture_notes/2010/fd_evolution_pdes_lnotes.pdf

[6] http://www.ehu.eus/aitor/irakas/fin/apuntes/pde.pdf

[7] https://www.sciencedirect.com/science/article/pii/S0377042713007085



"""

# ╔═╡ Cell order:
# ╠═3d6fb095-69d6-4345-b899-d50d07eb8955
# ╠═e2cd8d91-1281-44c4-a953-c45114174521
# ╟─013dd4ed-9eda-4d80-b1b1-2cf9cb2ffc68
# ╟─70f335de-b771-4c00-8785-56b2be24b6a1
# ╟─d61fb5c6-5e16-43a7-bea1-01d65a9c9c9f
# ╟─e0f8ed89-9ee2-44ca-89c0-9fa7760cbf8b
# ╟─c5fce78f-71c4-4be6-beeb-3c6788f6d088
# ╟─9a7ba6fb-d2d6-4e96-a4b4-3fd4337bd215
# ╠═610c2ee9-23f3-459e-8e7b-c25f88c4961f
# ╟─e2572c66-0256-46f4-90f4-ef31b57e95b4
# ╠═d75cdfa2-65a5-44b1-9595-6ce4c9555381
# ╠═574ac318-8d4d-4863-bd88-9d8dcb67eeb1
# ╟─f6db5bfa-1388-4704-bae7-81690c989594
# ╟─66162018-e21d-4e8f-b3f7-3aed8de06676
# ╟─fe04a936-e372-4adb-9724-c8da6df4d78e
# ╠═7bd3885d-a749-4047-9331-6aed47f5675c
# ╟─f697c6ba-5076-447a-ac06-28d6d0c3efe2
# ╠═98599ee3-4785-4015-96a9-38b892fdd1fb
# ╟─9b6c6ba6-e673-469a-bf4e-dff5a853d68b
# ╠═56510602-a7b2-430d-8a67-ad557746d2ac
# ╟─a3874dd6-cfe9-4abb-990a-d2550999a68e
# ╟─3a47e817-4f80-4750-82ac-e98d2f4008d2
# ╟─ee882691-14a9-4085-8a8a-01a60275915b
# ╠═e98a6e02-3af8-4366-80a0-0578c91f3353
# ╟─a619d0ed-ece3-40cb-b31b-d1b8e1712eec
# ╟─3ffe4d1a-15ce-4224-bccb-69dcb461cfbc
# ╠═5539e8e1-73df-4179-82d1-857c5c78a11d
# ╠═6e8c1180-ae88-44f0-b562-c79812ad3e06
# ╟─9c7e0eae-6697-421d-948e-853ce8cec511
# ╠═40bdbb79-5db0-40b1-bfa2-2d10e9c7c6a1
# ╠═b2395666-176c-4026-87e7-ac14bef0c234
# ╟─f7321227-4305-4f9a-9c12-abffea95b22a
# ╟─4907a8b6-a136-41df-9f3e-c538d50a3185
# ╠═4e4ce48b-ffe0-4ed8-be30-a3f1f2e3a207
# ╠═275db70e-b78d-4f72-a0a6-ee3356ac00b7
# ╟─4ae051ce-7431-414a-8e7e-fb6c2c02ae11
# ╠═a4895d9d-2b7e-4662-aabf-acdfd8527622
# ╟─6a58531a-93b9-478c-996e-314341886475
# ╟─58d9c9b7-533a-4418-9ab7-501da86450d9
# ╟─571459f0-55d7-41a5-9fbd-416ca8260f29
# ╟─90515297-fed3-404b-931b-36c179a7d0bc
# ╟─79533303-d400-403f-920e-9f75abd97330
# ╟─bc6f5299-7b9c-4a21-91e5-ce0105f6d318
# ╠═9312fad0-c2ce-437c-b1c2-8a68a6f77a83
# ╠═79438e59-c394-4011-b921-d8406becb7df
# ╟─f8f49dc1-5543-4956-9ce9-c39d399cc378
# ╟─431c651e-13e8-463b-9b1a-57e2b6337d1f
# ╠═e930562c-0041-45db-adcb-b038f5cd13d2
# ╟─aa272063-dda3-415d-a5a2-992bd468a690
# ╟─f6707b0a-940d-4954-85af-14cc9a305165
# ╠═c2da3bbf-05c0-42b8-81ce-9860d5cb5e87
# ╟─90c249fd-1255-42f0-81c9-3dcca77b1438
# ╟─4f1b8e9a-8529-49ac-aa8d-d385903b7477
# ╠═0d9774f3-7baa-4e82-a6ec-340b41095c5f
# ╟─3d2b47a9-dc84-40c6-b0c8-e7fe533d8856
# ╟─67eb179e-c78d-4c3e-9016-d4bb74ab47c1
# ╟─9bbf668b-edbb-48bd-9ad4-d68829c463ce
# ╟─15efbb34-ed94-4cf9-a660-00638436a246
