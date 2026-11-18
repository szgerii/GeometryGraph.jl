# Not using enums becuase I can't use them to index, or atleast they're jank
const LOG::UInt = 1
const INFO::UInt = 2
const WARN::UInt = 3
const ERR::UInt = 4

# Ascii only, for now
struct Cmsg
    type::UInt
    text::String
end

mutable struct RingBuffer{T}
    data::Vector{T}
    first::Int
    len::Int

    function RingBuffer{T}(capacity::Int) where {T}
        capacity > 0 || throw(ArgumentError("capacity must be positive"))
        new{T}(Vector{T}(undef, capacity), 1, 0)
    end
end

function Base.push!(b::RingBuffer, x)
    if b.len < length(b.data)
        b.len += 1
        b.data[mod1(b.first + b.len - 1, length(b.data))] = x
    else
        b.data[b.first] = x
        b.first = mod1(b.first + 1, length(b.data))
    end
    return b
end

function Base.empty!(b::RingBuffer)
    b.first = 1
    b.len = 0
    return b
end

Base.length(b::RingBuffer) = b.len
Base.isempty(b::RingBuffer) = b.len == 0
Base.eltype(::Type{RingBuffer{T}}) where {T} = T
Base.iterate(b::RingBuffer, i::Int = 0) = i >= b.len ? nothing : (@inbounds(b.data[mod1(b.first + i, length(b.data))]), i + 1)

global _messages::RingBuffer{Cmsg} = RingBuffer{Cmsg}(100)
global _filters::Array{Bool, 1} = [true,true,true,true]

# I would avoid the string creation if possible, thats why I recommend using the macros
macro log(msg, type=:(LOG))
    if isa(msg, Expr) || isa(msg, Symbol)
        msg = esc(msg)
    end
    return :(
        if (_filters[$type])
            push!(_messages, Cmsg($type,$msg))
        end
    )
end

function c_log(msg,type=LOG)
    if _filters[type]
        push!(_messages, msg)
    end
    return nothing
end

function clear_logs()
    empty!(_messages)
    return nothing
end