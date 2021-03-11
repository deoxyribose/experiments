using Gen
using Images
using Plots
using Random

Random.seed!(42);

tmp = load("/home/folzd/golem/tmp.png")

struct Pixel
    i::Int64
    j::Int64
end

struct Rectangle
    # tl = top-left
    # br = bottom-right
    tl::Pixel
    br::Pixel
end

function Pixel(vec::Vector{Int64})
    length(vec) != 2 && error("wrong length")
    return Pixel(vec[1], vec[2])
end

function Rectangle(vec::Vector{Pixel})
    length(vec) != 2 && error("wrong length")
    return Rectangle(vec[1], vec[2])
end

abstract type Node end

struct HorizontalNode <: Node
    left::Node
    right::Node
    rectangle::Rectangle
end

struct VerticalNode <: Node
    top::Node
    bottom::Node
    rectangle::Rectangle
end

struct LeafNode <: Node
    value::Float64
    rectangle::Rectangle
end

@gen function generate_segments(r::Rectangle)
    if @trace(bernoulli(0.6), :isleaf)
        value = @trace(normal(0, 1), :value)
        return LeafNode(value, r)
    else
        if @trace(bernoulli(0.5), :ishorizontal)
            frac = @trace(beta(2, 2), :frac)
            l = r.tl.j
            u = r.br.j
            mid  = round(l + (u - l) * frac)
            # don't confuse (i,j) with (x,y)
            # matrices are M[i, j], where 1 < i < nrows, 1 < j < ncols
            # coordinates are (x,y), where 1 < x < ncols, 1 < y < nrows
            # Pixels are matrix indices, not coordinates!
            left_br = Pixel(r.br.i,mid)
            right_tl = Pixel(r.tl.i,mid)
            left = @trace(generate_segments(Rectangle(r.tl,left_br)), :left)
            right = @trace(generate_segments(Rectangle(right_tl,r.br)), :right)
            return HorizontalNode(left, right, r)
        else
            frac = @trace(beta(2, 2), :frac)
            l = r.tl.i
            u = r.br.i
            mid  = round(l + (u - l) * frac)
            top_br = Pixel(mid,r.br.j)
            bottom_tl = Pixel(mid,r.tl.j)
            top = @trace(generate_segments(Rectangle(r.tl,top_br)), :top)
            bottom = @trace(generate_segments(Rectangle(bottom_tl,r.br)), :bottom)
            return VerticalNode(top, bottom, r)
        end
    end
end;


function render_rectangle(r::Rectangle)
    xs = [r.tl.j,r.br.j,r.br.j,r.tl.j]
    ys = [r.br.i,r.br.i,r.tl.i,r.tl.i]
    plot!(Shape(xs, ys), opacity=.1, yflip = true, legend = false)
end

function render_node(node::LeafNode)
    render_rectangle(node.rectangle)
end

function render_node(node::VerticalNode)
    render_node(node.top)
    render_node(node.bottom)
end;

function render_node(node::HorizontalNode)
    render_node(node.left)
    render_node(node.right)
end;

function render_segments_trace(trace)
    plot()
    node = get_retval(trace)
    render_node(node)
end;

function is_within_rectangle(r::Rectangle, i::Int64, j::Int64)
#    println(r)
#    println(i)
#    println(j)
    @assert i >= r.tl.i && i <= r.br.i && j >= r.tl.j && j <= r.br.j
end

# get_value_at searches a binary tree for
# the leaf node containing some value.
function get_value_at(i::Int64, j::Int64, node::LeafNode)
    is_within_rectangle(node.rectangle, i, j)
    return node.value
end

function get_value_at(i::Int64, j::Int64, node::HorizontalNode)
    is_within_rectangle(node.rectangle, i, j)
    if j <= node.left.rectangle.br.j
        get_value_at(i, j, node.left)
    else
        get_value_at(i, j, node.right)
    end
end

function get_value_at(i::Int64, j::Int64, node::VerticalNode)
    is_within_rectangle(node.rectangle, i, j)
    if i <= node.top.rectangle.br.i
        get_value_at(i, j, node.top)
    else
        get_value_at(i, j, node.bottom)
    end
end

@gen function screen_model_dynamic(size::Tuple{Int64,Int64})
    nrows, ncols = size
    screenshot = Array{Float64}(undef, nrows, ncols)
    #node = @trace(generate_segments(Rectangle(Pixel(1,nrows),Pixel(ncols,1))))
    node = @trace(generate_segments(Rectangle(Pixel(1,1),Pixel(nrows,ncols))))
    noise = @trace(gamma(1, 1), :noise)
    for i in 1:nrows
        for j in 1:ncols
            screenshot[i,j] = @trace(normal(get_value_at(i, j, node),noise), (:img, i, j))
        end
    end
    return screenshot
end

@gen function pixel_kernel(i::Int64, j::Int64, tree::Node, noise::Float64)
    pixel = @trace(normal(get_value_at(i, j, tree),noise), :pix)
    return pixel
end 

function get_index_matrices(matrix::Array{Float64})
    indices = CartesianIndices(matrix)
    is = convert(Array{Int64}, getindex.(indices,1))
    js = convert(Array{Int64}, getindex.(indices,2))
    return is, js
end

@gen (static) function screen_model(size::Tuple{Int64,Int64})
    screenshot = Array{Float64}(undef, size)
    nrows,ncols = size
    tree = @trace(generate_segments(Rectangle(Pixel(1,1),Pixel(nrows,ncols))), :tree)
    noise = @trace(gamma(1, 1), :noise)
    is, js = get_index_matrices(screenshot)
    screenshot = @trace(Map(pixel_kernel)(is,js,fill(tree, size), fill(noise, size)), :img)
    return reshape(screenshot,size)
end

function render_screen_trace(trace)
    Gray.(get_retval(trace))
end

function do_inference(model, img, amount_of_computation)
    # condition on image
    observations = choicemap()
    nrows, ncols = size(img)
    for i in 1:nrows
        for j in 1:ncols
            observations[(:img, i, j)] = img[i,j]
        end
    end
    # Call importance_resampling to obtain a likely trace consistent
    # with our observations.
    (trace, _) = importance_resampling(model, (size(img),), observations, amount_of_computation);
    return trace
end;


# generate segmentation
# trace = simulate(generate_segments, (Rectangle(Pixel(1,1), Pixel(648,1152)),))

# generate screen
# trace = simulate(screen_model, (size(tmp),))
# Gray.(get_retval(trace))

# inference
# do_inference(screen_model, tmp, 10000)

load_generated_functions()

# @profiler trace = simulate(screen_model, (size(tmp),))
